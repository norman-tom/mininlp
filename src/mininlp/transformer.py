from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoder():
    """Positional Encoder of token embedding.
    
    As Transformers are input position invariant, positional information
    must be added to the input vector. Positional Encoder provides this 
    positional information. Positional encoding as defined by 'Attention is 
    All You Need'.

    Parameters
    ----------
    max_seq : int 
        The maximum length input token sequence
    embedding_dim : int
        The embedding vector dimension 
    """

    SCALE = 10_000

    def __init__(self, max_seq: int, embedding_dim: int) -> None:
        self._encoding = torch.empty(1, max_seq, embedding_dim, requires_grad=False)
        self._encode(max_seq, embedding_dim)
    
    def _encode(self, max_seq: int, embedding_dim: int):
        for pos in range(max_seq):
            for i in range(embedding_dim // 2):
                self._encoding[:, pos, 2*i] = np.sin(
                    pos / self.SCALE ** (2 * i / embedding_dim)
                )
                self._encoding[:, pos, 2*i+1] = np.cos(
                    pos / self.SCALE ** (2 * i / embedding_dim)
                )
    
    @property
    def encoding(self) -> torch.Tensor:
        return self._encoding

class Embedding(nn.Module):
    """Token embedding plus positional encoding.

    Embedding learns the vector representation of each token. The embedding
    is a matrix of size vocabulary size x token embedding dimension. The positional
    encoding vector is added directly to the token embedding to create the input 
    vector to the encoder and decoder modules.

    Parameters
    ----------
    vocab_size : int
        size of the vocabulary determined during tokenization.

    embedding_dim : int 
        the dimesion of the embedding, aka the d_model in the paper "Attention is All You Need".

    max_seq: int
        Maximum sequence of input token for the encoder and decoder.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, max_seq: int) -> None:
        super().__init__()
        self._token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self._pos_encoding = PositionalEncoder(max_seq, embedding_dim)

    def forward(self, tkn_ids) -> torch.Tensor:
        # Token embeddings + positonal encoding broadcasted to all batch examples
        x = self._token_embedding(tkn_ids)
        x = x + self._pos_encoding.encoding.expand(x.size()).to(x.device)
        return x

class LanguageHead(nn.Module):
    """Predict the token class from the Decoder's output vector.
    """

    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self._projection = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._projection(x)

class FeedForward(nn.Module):
    """Feedforward module, skip connection, layer normalisation 
    and two fully connect layers with the inner layer dimmension a 
    factor of the outter layer.
    """

    def __init__(self, embedding_dim: int, factor: int = 4) -> None:
        super().__init__()
        self._laynorm = nn.LayerNorm(embedding_dim)
        self._dropout = nn.Dropout(0.2)
        self._ff = nn.Sequential(
            nn.Linear(embedding_dim, factor*embedding_dim),
            nn.ReLU(),
            nn.Linear(factor*embedding_dim, embedding_dim),
        )

    def forward(self, x) -> torch.Tensor:
        # Layer normalization before feedforward which is standard practice.
        x = self._laynorm(x)
        x = self._dropout(x)
        x = x + self._ff(x)
        return x

class Attention():
    """Computes the attention of Queries, Keys and Values
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, 
                 Q: torch.Tensor, 
                 K: torch.Tensor, 
                 V: torch.Tensor, 
                 mask=None) -> torch.Tensor:
        Kt = torch.transpose(K, -2, -1)
        score = torch.matmul(Q, Kt)
        key_dim = torch.tensor(K.size()[2])
        scale = score / torch.sqrt(key_dim)
        if mask is not None:
            mask[mask == 1] = torch.inf
            scale = scale - mask
        # softmax over K so last dimension
        return torch.matmul(F.softmax(scale, dim=-1), V) 

class MultiHeadAttention(nn.Module):
    """Computes the mulit-head attention of Queries, Keys, Values
    """

    def __init__(self, embedding_dim, num_heads) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._n_heads = num_heads
        self._att = Attention()
        self._projection = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(3)
        ])
        self._reprojection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, Q, K, V, mask=None) -> torch.Tensor:
        B, T, C = Q.size()
        Q = self._projection[0](Q).view(B, T, self._n_heads, C // self._n_heads).transpose(1, 2)
        K = self._projection[1](K).view(B, T, self._n_heads, C // self._n_heads).transpose(1, 2)
        V = self._projection[2](V).view(B, T, self._n_heads, C // self._n_heads).transpose(1, 2)
        x = self._att(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self._reprojection(x)
        return x

class Encoder(nn.Module):
    """Encoder module of the transformer architecture
    """

    def __init__(self, embedding_dim, num_heads, factor) -> None:
        super().__init__()
        self._mha = MultiHeadAttention(embedding_dim, num_heads)
        self._ff = FeedForward(embedding_dim, factor)
        self._laynorm = nn.LayerNorm(embedding_dim)

    def forward(self, x) -> torch.Tensor:
        x = self._laynorm(x)
        x = x + self._mha(x, x, x)
        x = self._ff(x)
        return x

class Decoder(nn.Module):
    """Decoder module of the transformer architecture
    """

    def __init__(self, embedding_dim: int, num_heads: int, factor: int) -> None:
        super().__init__()
        self._laynorm1 = nn.LayerNorm(embedding_dim)
        self._laynorm2 = nn.LayerNorm(embedding_dim)
        self._mmha = MultiHeadAttention(embedding_dim, num_heads)
        self._mha = MultiHeadAttention(embedding_dim, num_heads)
        self._ff = FeedForward(embedding_dim, factor)
    
    def forward(self, x: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self._laynorm1(x)
        x = x + self._mmha(x, x, x, mask)
        if z is not None: 
            # If there is cross attention
            x = self._laynorm2(x)
            x = x + self._mha(z, z, x)
        x = self._ff(x)
        return x

class DTransformer(nn.Module):
    """Decoder only Transformer. 

    Decoder only Transformers are used for sequence modeling tasks,
    such as text generation. Given a dataset the goal is to learn a probability  
    distribution of the next token given the previous tokens in the sequence.

    Parameters
    ----------
    N : int
        Number of decoder modules
    embedding_dim : int
        The dimesion of the embedding vector
    vocab_size : int
        The size of the vocabulary
    max_seq : int
        The maximum input sequence of tokens
    num_heads : int
        The number of multihead attention heads
    factor : int
        The multiplcation factor to determine the dimesion of the internal 
        feedforward layer. Multiple of the embedding dimension.
    mask : int | None
        The masking of input sequence tokens.
    """

    def __init__(self, 
                 N: int, 
                 embedding_dim:int, 
                 vocab_size: int, 
                 max_seq: int,
                 num_heads: int,
                 factor: int,
                 mask: torch.Tensor=None) -> None:
        super().__init__()
        self._embedding = Embedding(vocab_size, embedding_dim, max_seq)
        self._decoders = nn.ModuleList(Decoder(embedding_dim, num_heads, factor) for _ in range(N))
        self._lang_head = LanguageHead(embedding_dim, vocab_size)
        self._mask = mask
        self._N = N
        self._seq_len = max_seq

    def forward(self, tkn_ids) -> torch.Tensor:
        x = self._embedding(tkn_ids) 
        for decoder in self._decoders:
            x = decoder(x, None, self._mask)
        x = self._lang_head(x)
        return x
    
    def generate(self, prompt: torch.Tensor, n: int) -> torch.Tensor:
        """Generator for the DTransformer. 

        Parameters
        ----------
        prompt : torch.Tensor
            The sequence of token ids to start the text generation.
        n: int
            The number of tokens to generate.
        
        Returns
        -------
        torch.Tensor
            The generated sequence of token ids.
        """
        buffer = torch.empty(0, dtype=torch.long, device=prompt.device)
        prompt = prompt[:, -self._seq_len:]  # Truncate the prompt to the max sequence length.
        for _ in range(n):
            o = self(prompt)                # Predict the sequence
            o = F.softmax(o, dim=-1)        # Softmax the output (probabilities of the next token
            o = torch.multinomial(o.squeeze(), 1)[-1] # Sample from vocabulary using pytorch multinomial
            prompt = torch.cat([prompt[:,1:], o[None]], dim=1) # Update the prompt with the predicted token
            buffer = torch.cat([buffer, o], dim=0) # Append the predicted token to the buffer
        return buffer
    
class ETransformer():
    """Encoder only Transformer
    """
    pass

class EDTransformer():
    """Encoder Decoder Transformer for sequence to sequence prediction.
    """
    pass