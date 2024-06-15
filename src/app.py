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

MODEL_NAME = 'decoder_transformer_v0.1'
SEQ_LEN = 1024
EMBEDDING_DIM = 512
HEADS = 8
LAYERS = 6
FACTOR = 4
BATCH_SIZE = 32
EPOCHS = 1
N_DATASET = 200_000
LR = 1e-4
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
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(MODEL_PATH, "tokenizer.pkl"))

    # Make tokens a factor of 8, increase speed. 
    for i in range(len(tokenizer), 128):
        tokenizer._tokens[i] = ["UNK"]

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
    
    # torch compile see https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html not supported on windows
    # model = torch.compile(model)

    if resume is not None:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, resume + '.pt')))

    print(f'Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    model.train()
    import time;
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), config['lr'])
    epochs = range(1)
    for _ in epochs:
        n_batch = 0
        epoch_loss = 0.0
        for x, y in data_loader:
            #start of batch
            start = time.time()

            # forward and backwards pass
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            # mixed percision see https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                o = model(x) 
                loss = criterion(o.transpose(1,2), y.long())

            loss.backward()
            optimizer.step()
            n_batch += 1
            epoch_loss += loss.item()

            # print training metrics
            torch.cuda.synchronize()
            dt = time.time() - start
            print(f"Batch: {n_batch} \t Loss: {loss.item():.4f} \t dt: {dt * 1e3:.2f} \t tkn\s: {BATCH_SIZE * SEQ_LEN/(dt):.2f}")

    #training.train(model, data_loader, criterion, config['lr'], config['epochs'])
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, config['model_name'] + '.pt'))

if __name__ == "__main__":
    train(config['pre_trained'])