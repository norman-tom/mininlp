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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "Dtransformer.pt")
SEQ_LEN = 64
EMBEDDING_DIM = 128
HEADS = 4
LAYERS = 4
FACTOR = 4

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    text = open(os.path.join("data", "anna.txt")).read()
    vocabulary = set(text)
    vocabulary.add("<sos>")
    vocabulary.add("<eos>")
    tokenizer = Tokenizer(vocabulary)
    data = SequenceDataset(text, SEQ_LEN, tokenizer)
    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).to(device)
    model = DTransformer(LAYERS, EMBEDDING_DIM, len(vocabulary), SEQ_LEN, HEADS, FACTOR, mask).to(device)
    data_loader = DataLoader(data, 1024)
    criterion = nn.CrossEntropyLoss()
    training.train(model, data_loader, criterion, 5e-4, 100)
    tokenizer.save(os.path.join(ROOT, "models", "vocabulary"))
    torch.save(model.state_dict(), MODEL_PATH)

def inference():
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(ROOT, "models", "vocabulary.pkl"))
    model = DTransformer(LAYERS, EMBEDDING_DIM, len(tokenizer), SEQ_LEN, HEADS, FACTOR, None)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device="cpu")))
    prompt = "Stepan Arkadyevitch was a truthful"
    prompt = tokenizer.encode(prompt)
    prompt = F.pad(prompt, pad=(SEQ_LEN - len(prompt), 0), mode='constant', value=tokenizer._token_ids["<sos>"])
    model.eval()
    o = model.generate(prompt[None])
    print(*tokenizer.decode(o[0]))


if __name__ == "__main__":
    train()
    inference()