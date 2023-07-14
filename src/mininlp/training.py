import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

def train(model: nn.Module , data: DataLoader, criterion, lr, n_epochs):
    device = next(model.parameters()).device
    optimizer = Adam(model.parameters(), lr)
    epochs = tqdm(range(n_epochs))
    for _ in epochs:
        n_batch = 0.0
        epoch_loss = 0.0
        for x, y in data:
            N = x.size()[0] * x.size()[1]
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            o = model(x)
            # Each sequence predicting the next token in that sequence for all positions, across all batches. 
            loss = criterion(o.view(N, -1), y.view(N))  
            loss.backward()
            n_batch += 1
            epoch_loss += loss.item()
            optimizer.step()
        epoch_loss /= n_batch
        epochs.set_postfix(loss=f"{epoch_loss: .4f}")