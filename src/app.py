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


MODEL_NAME = 'Dtransformer_large'
SEQ_LEN = 128
EMBEDDING_DIM = 512
HEADS = 8
LAYERS = 4
FACTOR = 4
BATCH_SIZE = 64

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models")

def train(resume=None):
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    text = open(os.path.join("data", "anna.txt")).read()

    if resume is None:
        vocabulary = set(text)
        vocabulary.add("<sos>")
        vocabulary.add("<eos>")
        tokenizer = Tokenizer(vocabulary)
    else:
        tokenizer = Tokenizer()
        tokenizer.load(os.path.join(MODEL_PATH, 'vocabulary_large.pkl'))

    data = SequenceDataset(text, SEQ_LEN, tokenizer)
    data_loader = DataLoader(data, BATCH_SIZE)

    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).to(device)
    model = DTransformer(LAYERS, EMBEDDING_DIM, len(tokenizer), SEQ_LEN, HEADS, FACTOR, mask).to(device)
    if resume is not None:
        model.load_state_dict(os.path.join(MODEL_PATH, MODEL_NAME + '.pt'))

    criterion = nn.CrossEntropyLoss()
    training.train(model, data_loader, criterion, 1e-4, 20)

    tokenizer.save(os.path.join(ROOT, "models", "vocabulary_large"))
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME + '.pt'))

def inference():
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(ROOT, "models", "vocabulary_large.pkl"))
    model = DTransformer(LAYERS, EMBEDDING_DIM, len(tokenizer), SEQ_LEN, HEADS, FACTOR, None)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device="cpu")))
    prompt = "Stepan Arkadyevitch was a truthful"
    prompt = tokenizer.encode(prompt)
    prompt = F.pad(prompt, pad=(SEQ_LEN - len(prompt), 0), mode='constant', value=tokenizer._token_ids["<sos>"])
    model.eval()
    o = model.generate(prompt[None])
    print(*tokenizer.decode(o[0]))


if __name__ == "__main__":
    train('Dtransformer_large')
    inference()
