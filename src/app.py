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


MODEL_NAME = 'decoder_transformer_v1'
SEQ_LEN = 128
EMBEDDING_DIM = 512
HEADS = 8
LAYERS = 4
FACTOR = 4
BATCH_SIZE = 256

config = {
    "model_name": MODEL_NAME,
    "seq_len": SEQ_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "heads": HEADS,
    "layers": LAYERS,
    "factor": FACTOR,
    "batch_size": BATCH_SIZE
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models")

json.dump(config, open(os.path.join(MODEL_PATH, MODEL_NAME + '.json'), 'w'))

def train(resume=None):
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(MODEL_PATH, "tokenizer.pkl"))

    data = SequenceDataset("./data/anna.txt", tokenizer, SEQ_LEN, 50000)
    data_loader = DataLoader(data, BATCH_SIZE)

    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).to(device)
    model = DTransformer(LAYERS, EMBEDDING_DIM, len(tokenizer), SEQ_LEN, HEADS, FACTOR, mask).to(device)

    if resume is not None:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '.pt')))

    print(f'Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    model.train()
    criterion = nn.CrossEntropyLoss()
    training.train(model, data_loader, criterion, 1e-4, 50)

    torch.save(model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME + '.pt'))


if __name__ == "__main__":
    train()
