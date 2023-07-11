import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (Dataset, DataLoader)
from mininlp.transformer import DTransformer
import mininlp.training as training
from mininlp.data import SequenceDataset
import os

class TestTraining(unittest.TestCase):
    def test_simple_train(self):
        max_seq = 10
        de = 24
        v_size = 26
        n_heads = 4
        N = 2
        factor = 4
        mask = torch.triu(torch.ones(max_seq, max_seq), diagonal=1)
        model = DTransformer(N, de, v_size, max_seq, n_heads, factor, mask)

        class Data(Dataset):
            def __init__(self, vocab_size, n_examples, l_seq) -> None:
                super().__init__()
                self._data = torch.randint(0, vocab_size, (n_examples, l_seq+1))
                self._vocab_size = vocab_size
                self._seq = l_seq

            def __len__(self):
                return self._data.size()[0]
            
            def __getitem__(self, index) -> torch.Tensor:
                x = self._data[index][:-1]
                y = F.one_hot(self._data[index][1:], num_classes=self._vocab_size)
                y = y.float()
                return x, y
        
        examples = 1000
        dataset = Data(v_size, examples, max_seq)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        lr = 0.001
        n_epochs = 200
        training.train(model, dataloader, criterion, lr, n_epochs=n_epochs)
    
    def test_token_train(self):
        max_seq = 20
        de = 128
        n_heads = 4
        N = 2
        factor = 4
        data = SequenceDataset(os.path.join("data", "anna.txt"), max_seq)
        v_size = len(data._vocabulary)
        mask = torch.triu(torch.ones(max_seq, max_seq), diagonal=1)
        model = DTransformer(N, de, v_size, max_seq, n_heads, factor, mask)
        data_loader = DataLoader(data)
        lr = 0.0001
        n_epochs = 1
        criterion = nn.CrossEntropyLoss()
        training.train(model, data_loader, criterion, lr, n_epochs)