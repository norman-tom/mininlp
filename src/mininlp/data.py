from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import pickle
import string


def assci_tokens():
    tokens = [c for c in ["<sos>", "<eos>", "<pad>", "<unk>", "<mask>"]]
    tokens += list(string.printable)
    tokens = set(tokens)
    return tokens

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
            self._tokens = {i: c for c, i in self._token_ids.items()}

    def __len__(self):
        return len(self._tokens)
    
    def decode(self, ids: torch.Tensor) -> list[str]:
        """Decode token ids to tokens.
        
        Parameters
        ----------
        ids : torch.Tensor | int
            The token ids to decode.
        
        Returns
        -------
        list[str]
            The decoded tokens.
        """

        if isinstance(ids, list):
            ids = torch.tensor(ids)

        if ids.dim() == 0:
            ids = ids.unsqueeze(0)

        return [self._tokens[t.item()] for t in ids]
    
    def encode(self, tokens: list[str]) -> torch.Tensor:
        """Encode tokens to token ids.
        
        Parameters
        ----------
        tokens : list[str] | str
            The tokens to encode.
        
        Returns
        -------
        torch.Tensor
            The encoded token ids.
        """
        
        if isinstance(tokens, str):
            tokens = [t for t in tokens]

        return torch.tensor([self._token_ids[t] for t in tokens], dtype=torch.int32)
    
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

    def tokenize_document(self, path: str) -> torch.Tensor:
        """Given a string of text, convert it to a list of tokens."""

        with open(path) as f:
            text = f.read()
            return self.encode(text)
   
class SequenceDataset(Dataset):
    """Dataset stores the samples and their corresponding labels.
    """

    def __init__(self, document_path: str, tokenizer: Tokenizer, context_len: int, number_examples: int) -> None:
        self._tokenizer = tokenizer
        self._context_len = context_len
        self._dataset = self.generate_dataset(document_path, tokenizer, context_len, number_examples)

    def __len__(self):
        return len(self._dataset)


    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dataset[index]
    
    def generate_dataset(
            self,
            document_path: str, 
            tokenizer: Tokenizer, 
            context_length: int, 
            num_examples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate a dataset of training examples.
        """

        encoded_document = self._tokenizer.tokenize_document(document_path)
        examples = self.generate_example_set(encoded_document, tokenizer, context_length, num_examples)
        return examples

    def generate_example_set(
            self, 
            encoded_document: torch.Tensor, 
            tokenizer: Tokenizer, 
            context_length: int, 
            num_examples: int) -> torch.Tensor:
        """Given and encoded documnent, randomly generate a set of training examples.
        """
        
        indices = torch.randint(0, len(encoded_document) - context_length, (num_examples,))
        examples = [self.process_example(encoded_document[i: i+context_length], tokenizer, context_length) for i in indices]
        return examples
    
    def pad_sequence(self, 
                     sequence: torch.Tensor, 
                     length: int) -> torch.Tensor:
        """Pad a sequence with the padding token to a given length.
        """

        tensor = torch.full((length - len(sequence),), self._tokenizer._token_ids["<pad>"], dtype=torch.int32)
        return torch.cat((tensor, sequence))

    def process_example(
            self, 
            example: str, 
            tokenizer: Tokenizer, 
            context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a training example from a document.
        """
        
        assert len(example) == context_length

        tokens_to_get = int(torch.rand(1) * context_length + 1)        # torch.rand is the interval [0, 1), will not return 1 so + 1 for full range. 
        label = example[:tokens_to_get]
        feature = torch.cat((tokenizer.encode(['<sos>']), label[:-1])) # there will always be space for the start of sentence token.
        feature = self.pad_sequence(feature, context_length)
        label = self.pad_sequence(label, context_length)

        return feature, label