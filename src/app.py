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
LAYERS = 6
FACTOR = 4
BATCH_SIZE = 96

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

    text = open(os.path.join("data", "anna.txt")).read()

    if resume is None:
        vocabulary = set(text)
        vocabulary.add("<sos>")
        vocabulary.add("<eos>")
        tokenizer = Tokenizer(vocabulary)
        tokenizer.save(os.path.join(MODEL_PATH, f'vocab_{MODEL_NAME}'))
    else:
        tokenizer = Tokenizer()
        tokenizer.load(os.path.join(MODEL_PATH, f'vocab_{MODEL_NAME}.pkl'))

    data = SequenceDataset(text, SEQ_LEN, tokenizer)
    data_loader = DataLoader(data, BATCH_SIZE)

    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).to(device)
    model = DTransformer(LAYERS, EMBEDDING_DIM, len(tokenizer), SEQ_LEN, HEADS, FACTOR, mask).to(device)

    if resume is not None:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '.pt')))

    print(f'Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    model.train()
    criterion = nn.CrossEntropyLoss()
    training.train(model, data_loader, criterion, 1e-5, 20)

    torch.save(model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME + '.pt'))

def inference():
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(MODEL_PATH, f'vocab_{MODEL_NAME}.pkl'))
    model = DTransformer(LAYERS, EMBEDDING_DIM, len(tokenizer), SEQ_LEN, HEADS, FACTOR, None)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '.pt'), map_location=torch.device(device="cpu")))
    model.eval()
    prompt = "Stepan Arkadyevitch had not chosen his political opinions or his views"
    prompt = tokenizer.encode(prompt)
    prompt = F.pad(prompt, pad=(SEQ_LEN - len(prompt), 0), mode='constant', value=tokenizer._token_ids["<sos>"])
    model.eval()
    o = model.generate(prompt[None])
    print(*tokenizer.decode(o[0]))


if __name__ == "__main__":
    #train(None)
    inference()
