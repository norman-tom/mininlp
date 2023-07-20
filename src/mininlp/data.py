from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import pickle
import os

class Tokenizer():
    """Character level tokenizer.
    
    Parameters
    ----------
    vocabulary : set, optional
        The vocabulary to use for tokenization. If None, the tokenizer will be empty and the state must be loaded.
    """

    def __init__(self, vocabulary: set=None) -> None:
        self._token_ids = None
        self._tokens = None
        # If vocabulary is not None, create the token ids and tokens
        if vocabulary is not None:
            self._token_ids = {c: i for i, c in enumerate(vocabulary)}
            self._tokens = {i: c for i, c in enumerate(self._token_ids)}
        self.decode = lambda ids: [self._tokens[t.item()] for t in ids]
        self.encode = lambda tokens: torch.tensor([self._token_ids[c] for c in tokens])

    def __len__(self):
        return len(self._token_ids)
    
    def save(self, path: str) -> None:
        """Save the tokenizer state to a file.
        
        Parameters
        ----------
        path : str
            The path to save the tokenizer file.
        """

        tokens = {"tokens": self._tokens, "token_ids": self._token_ids}
        pickle.dump(tokens, open(path + ".pkl", "wb"))
        with open(path + ".txt", "w") as f:
            f.write("\n".join(self._tokens.values()))

    def load(self, path: str) -> None:
        """Load the tokenizer state from a file.
        
        Parameters
        ----------
        path : str
            The path to the tokenizer file.
        """

        tokens = pickle.load(open(path, "rb"))
        self._tokens = tokens["tokens"]
        self._token_ids = tokens["token_ids"]

class SequenceDataset(Dataset):
    def __init__(self, raw: str, max_seq: int, tokenizer: Tokenizer) -> None:
        super().__init__()
        self._raw: str = raw
        self._tokenizer: Tokenizer = tokenizer
        self._sequence: str = self._tokenizer.encode(self._raw)
        self._max_seq: int = max_seq

    def __len__(self):
        return len(self._sequence) - self._max_seq

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._sequence[index:index + self._max_seq - 1]
        x = torch.cat([torch.tensor([self._tokenizer.encode(["<sos>"])]), x], dim=0)
        y = self._sequence[index:index+self._max_seq]
        return x, y