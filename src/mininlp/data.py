from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import pickle

class Tokenizer():
    def __init__(self, vocabulary: set) -> None:
        self._vocabulary = vocabulary
        self._token_ids = {c: i for i, c in enumerate(self._vocabulary)}
        self._tokens = {i: c for i, c in enumerate(self._token_ids)}
        self.decode = lambda ids: [self._tokens[t.item()] for t in ids]
        self.encode = lambda tokens: torch.tensor([self._token_ids[c] for c in tokens])

    def __len__(self):
        return len(self._vocabulary)
    
    def save(self, path: str) -> None:
        pickle.dump(self._vocabulary, open(path, "wb"))

class SequenceDataset(Dataset):
    def __init__(self, path: str, max_seq: int, tokenizer: Tokenizer) -> None:
        super().__init__()
        self._raw: str = open(path).read()
        self._tokenizer: Tokenizer = tokenizer
        self._sequence: str = self._tokenizer.encode(self._raw)
        self._max_seq: int = max_seq

    def __len__(self):
        return len(self._raw) - self._max_seq

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._sequence[index:index + self._max_seq - 1]
        x = torch.cat([torch.tensor([self._token_ids["<sos>"]]), x], dim=0)
        y = self._sequence[index:index+self._max_seq]
        return x, y