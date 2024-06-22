import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mininlp.transformer import DTransformer
from mininlp.data import SequenceDataset
from mininlp import training
from mininlp.data import Tokenizer
import json
import sys
import time

MODEL_NAME = 'decoder_transformer_v0.2'
SEQ_LEN = 1024
EMBEDDING_DIM = 512     #flash attention can only handle head size of 64
HEADS = 8
LAYERS = 8
FACTOR = 4
BATCH_SIZE = 512
MINI_BATCH = 32
EPOCHS = 1
N_DATASET = 900_000
LR = 4e-4
PRE_TRAINED = None

config = {
    "model_name": MODEL_NAME,
    "seq_len": SEQ_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "heads": HEADS,
    "layers": LAYERS,
    "factor": FACTOR,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "n_dataset": N_DATASET,
    "lr": LR,
    "batch_size": BATCH_SIZE,
    "pre_trained": PRE_TRAINED
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models")

json.dump(config, open(os.path.join(MODEL_PATH, config['model_name'] + '.json'), 'w'))

def train(resume=None):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(MODEL_PATH, "tokenizer.pkl"))

    data = SequenceDataset("./data/anna.txt", tokenizer, config['seq_len'], config['n_dataset'])
    data_loader = DataLoader(data, config['batch_size'], shuffle=False, pin_memory=True)

    # Free speed up with tensor float32 matmul, no so great on 3070
    # torch.set_float32_matmul_precision('high')

    model = DTransformer(
        config['layers'], 
        config['embedding_dim'], 
        len(tokenizer), 
        config['seq_len'], 
        config['heads'], 
        config['factor'], 
        True).to(device)

    # Resume from a previously trained model.
    if resume is not None:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, resume + '.pt')))
    
    # torch compile see https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html not supported on windows
    if sys.platform.startswith('linux'):
        model = torch.compile(model)

    print(f'Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), config['lr'])
    epochs = range(config['epochs'])
    n_batch = 0
    for _ in epochs:
        epoch_loss = 0.0
        for x, y in data_loader:
            #start of batch
            start = time.time()
            # loss for batch
            batch_loss = 0.0
            # device
            x, y = x.to(device), y.to(device)
            # forward and backwards pass
            
            #with torch.autocast(device_type=device, dtype=torch.bfloat16):
            #    p = model(x)
            #    l = criterion(p.transpose(1,2), y.long())
            #l.backward()
            #model_grad = model._lang_head._projection.weight.grad
            #print(model_grad[0][:5])
            #optimizer.zero_grad()

            # gradient accumulation 
            num_mini_batches = torch.ceil(torch.tensor(x.shape[0] / MINI_BATCH)).int()
            for i in range(num_mini_batches):
                x_mini = x[i*MINI_BATCH:(i+1)*MINI_BATCH]
                y_mini = y[i*MINI_BATCH:(i+1)*MINI_BATCH]
                # mixed percision see https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    o = model(x_mini)
                    loss = criterion(o.transpose(1,2), y_mini.long()) / num_mini_batches
                    #loss = F.cross_entropy(o.view(-1, o.size(-1)), y_mini.long().view(-1)) / num_mini_batches
                loss.backward()
                batch_loss += loss.item()

                #model_grad = model._lang_head._projection.weight.grad
                #print(model_grad[0][:5])
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            n_batch += 1
            epoch_loss += batch_loss

            # print training metrics
            torch.cuda.synchronize()
            dt = time.time() - start
            print(f"Batch: {n_batch} \t Loss: {batch_loss:.4f} \t dt: {dt * 1e3:.2f} \t tkn\s: {x.shape[0] * SEQ_LEN/(dt):.2f}")

    #training.train(model, data_loader, criterion, config['lr'], config['epochs'])
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, config['model_name'] + '.pt'))

if __name__ == "__main__":
    train(config['pre_trained'])