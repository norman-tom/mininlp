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


MODEL_NAME = 'decoder_transformer_v3.0'
SEQ_LEN = 512
EMBEDDING_DIM = 768
HEADS = 8
LAYERS = 8
FACTOR = 4
BATCH_SIZE = 256
EPOCHS = 1
N_DATASET = 10_000_000
PRE_TRAINED = None
LR = 1e-4

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
    "pre_trained": PRE_TRAINED,
    "lr": LR,
    "batch_size": BATCH_SIZE
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models")

json.dump(config, open(os.path.join(MODEL_PATH, config['model_name'] + '.json'), 'w'))

def train(resume=None):
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(MODEL_PATH, "tokenizer.pkl"))

    data = SequenceDataset("./data/anna.txt", tokenizer, config['seq_len'], config['n_dataset'])
    data_loader = DataLoader(data, config['batch_size'], shuffle=False, pin_memory=True)

    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).to(device)
    mask = None

    model = DTransformer(
        config['layers'], 
        config['embedding_dim'], 
        len(tokenizer), 
        config['seq_len'], 
        config['heads'], 
        config['factor'], 
        mask).to(device)

    if resume is not None:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, resume + '.pt')))

    print(f'Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    model.train()
    criterion = nn.CrossEntropyLoss()
    training.train(model, data_loader, criterion, config['lr'], config['epochs'])

    torch.save(model.state_dict(), os.path.join(MODEL_PATH, config['model_name'] + '.pt'))


if __name__ == "__main__":
    train(config['pre_trained'])
