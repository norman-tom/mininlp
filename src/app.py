import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from mininlp.transformer import DTransformer
from mininlp.data import SequenceDataset
from mininlp import training

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "Dtransformer.pt")
SEQ_LEN = 64
EMBEDDING_DIM = 128
HEADS = 4
LAYERS = 4
FACTOR = 4

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    data = SequenceDataset(os.path.join("data", "anna.txt"), SEQ_LEN)
    v_size = len(data._vocabulary)
    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).to(device)
    model = DTransformer(LAYERS, EMBEDDING_DIM, v_size, SEQ_LEN, HEADS, FACTOR, mask).to(device)
    data_loader = DataLoader(data, 1024)
    criterion = nn.CrossEntropyLoss()
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    training.train(model, data_loader, criterion, 5e-4, 200)
    pickle.dump(data._vocabulary, open(os.path.join(ROOT, "models", "vocabulary.pkl"), "wb"))
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()