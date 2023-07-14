from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class SequenceDataset(Dataset):
    def __init__(self, path, l_seq) -> None:
        super().__init__()
        self._raw = open(path).read()
        self._vocabulary = set(self._raw)
        self._vocabulary.add("<sos>")
        self._token_ids = {c: i for i, c in enumerate(self._vocabulary)}
        self._tokens = {i: c for i, c in enumerate(self._token_ids)}
        self.token_decoder = lambda x: [self._tokens[i.item()] for i in x]
        self.token_encoder = lambda x: torch.tensor([self._token_ids[c] for c in x])
        self._sequence = self.token_encoder(self._raw)
        self._l_seq = l_seq

    def __len__(self):
        return len(self._raw) - self._l_seq

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._sequence[index:index + self._l_seq - 1]
        x = torch.cat([torch.tensor([self._token_ids["<sos>"]]), x], dim=0)
        y = self._sequence[index:index+self._l_seq]
        return x, y