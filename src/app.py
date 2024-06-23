import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mininlp.transformer import DTransformer
from mininlp.data import SequenceDataset
#rom mininlp import training
from mininlp.data import Tokenizer
import json
import sys
import time
from torch.utils.tensorboard import SummaryWriter

"""
TODO:
"""

MODEL_NAME = 'decoder_transformer_v0.2'
SEQ_LEN = 512
EMBEDDING_DIM = 256     
HEADS = 8             #flash attention can only handle head size of 64
LAYERS = 16
FACTOR = 4
BATCH_SIZE = 512
MINI_BATCH = 32
EPOCHS = 1
N_DATASET = 6_000_000 // BATCH_SIZE * BATCH_SIZE  # divisible by batch size to keep from recompiling
LR = 200e-6
LR_MIN = LR * 10e-2
LR_WU = 0.05
LR_END = 0.8

PRE_TRAINED = 'decoder_transformer_v0.1'

config = {
    "model_name": MODEL_NAME,
    "seq_len": SEQ_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "heads": HEADS,
    "layers": LAYERS,
    "factor": FACTOR,
    "batch_size": BATCH_SIZE,
    "mini_batch": MINI_BATCH,
    "epochs": EPOCHS,
    "n_dataset": N_DATASET,
    "lr_max": LR,
    "lr_min": LR_MIN,
    "lr_wu": LR_WU,
    "lr_end": LR_END,
    "pre_trained": PRE_TRAINED
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models")
json.dump(config, open(os.path.join(MODEL_PATH, config['model_name'] + '.json'), 'w'))
writer = SummaryWriter("runs/transformer_v0.2")

def lr_schedule(batch, max_lr, min_lr, n_batchs, pct_start, pct_end):
    # cosine learning rate
    warmup_step = torch.tensor(pct_start * n_batchs)
    bottom_step = torch.tensor(pct_end * n_batchs)
    if batch < warmup_step:
        return (max_lr - min_lr) / warmup_step * batch + min_lr
    elif batch < bottom_step:
        # shift, scale, freq the cosine function
        return (max_lr - min_lr) / 2 * torch.cos(torch.pi / (bottom_step - warmup_step) * (batch - warmup_step)) + (max_lr + min_lr) / 2 
    else:
        return min_lr

def lr_update(optimizer, batch):
    lr = lr_schedule(
            batch, 
            config['lr_max'], 
            config['lr_min'], 
            config['epochs'] * config['n_dataset'] // config['batch_size'], 
            config['lr_wu'], 
            config['lr_end']
        )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(resume=None):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(MODEL_PATH, "tokenizer.pkl"))

    data = SequenceDataset("./data/anna.txt", tokenizer, config['seq_len'], config['n_dataset'])
    eval_idx = torch.randperm(len(data))[:config['batch_size']]
    eval_data = [data[i] for i in eval_idx]
    train_data = [data[i] for i in range(len(data)) if i not in eval_idx]
    eval_loader = DataLoader(eval_data, config['batch_size'], shuffle=False, pin_memory=True)
    data_loader = DataLoader(train_data, config['batch_size'], shuffle=False, pin_memory=True)

    # Free speed up with tensor float32 matmul, using bfloat16
    torch.set_float32_matmul_precision('high')

    model = DTransformer(
        config['layers'], 
        config['embedding_dim'], 
        len(tokenizer), 
        config['seq_len'], 
        config['heads'], 
        config['factor'], 
        True).to(device)
    
    print(f'Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad):}')
    
    # torch compile see https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html not supported on windows
    # to get torch.compile to work, update to latest nvidia drivers, use cuda 12.4 and pytorch nightly. 
    if sys.platform.startswith('linux') and True:
        model = torch.compile(model)

    # Resume from a previously trained model.
    if resume is not None:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, resume + '.pt')))

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), config['lr_min'])
    epochs = range(config['epochs'])
    batch = 0
    for e in epochs:
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
            lr = lr_update(optimizer, batch)
            optimizer.step()
            optimizer.zero_grad()
            batch += 1
            epoch_loss += batch_loss

            # print training metrics
            torch.cuda.synchronize()
            dt = time.time() - start
            print(f"Batch: {batch} \t Loss: {batch_loss:.4f} \t lr: {lr:.4e} \t dt: {dt * 1e3:.2f} \t tkn\s: {x.shape[0] * SEQ_LEN/(dt):.2f}")
            writer.add_scalar("Loss/train", batch_loss, batch)
            writer.add_scalar("param/lr", lr, batch)

            if batch % 100 == 0:
                eval_loss = 0.
                for i, (eval_x, eval_y) in enumerate(eval_loader):
                    eval_x, eval_y = eval_x.to(device), eval_y.to(device)
                    with torch.no_grad():
                        eval_o = model(eval_x)
                        eval_loss += F.cross_entropy(
                            eval_o.view(-1, 
                            eval_o.size(-1)), 
                            eval_y.long().view(-1))
                print(f"Eval Loss: {eval_loss.item()/(i+1):.4f}")
                writer.add_scalar("Loss/eval", eval_loss.item(), batch)

        #training.train(model, data_loader, criterion, config['lr'], config['epochs'])
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, config['model_name'] + '.pt'))

if __name__ == "__main__":
    train(None)