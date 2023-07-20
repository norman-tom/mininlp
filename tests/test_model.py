import unittest
import torch
from mininlp.transformer import (PositionalEncoder, 
                                 LanguageHead,
                                 Embedding,
                                 FeedForward,
                                 Attention,
                                 MultiHeadAttention,
                                 Encoder,
                                 Decoder,
                                 DTransformer)
from mininlp.data import SequenceDataset
import os

class TestModel(unittest.TestCase):
    #TODO Turns these into automated tests

    def test_positional_encoding(self):
        max_seq = 4
        embedding_dim = 4
        pos_encoder = PositionalEncoder(max_seq, embedding_dim)
        encoding = pos_encoder.encoding
        encoding

    def test_embedding(self):
        max_seq = 4
        embedding_dim = 4
        vocab_size = 6
        embedding = Embedding(vocab_size, embedding_dim, max_seq)
        x = embedding(torch.tensor([[[0, 0, 1, 2], [0, 0, 1, 2], [0, 0, 1, 2]]]))
        x

    def test_feedforward(self):
        embedding_dim = 4
        ff = FeedForward(embedding_dim)
        x = torch.rand(2, 4, 4)
        x = ff(x)
        x
    
    def test_attention(self):
        embedding_dim = 4
        max_seq = 6
        attention = Attention()
        # Rolling with self attention
        Q = torch.rand(1, max_seq, embedding_dim)
        K = Q.clone()
        V = Q.clone()
        mask = torch.triu(torch.ones(1, max_seq, max_seq), diagonal=1)
        score = attention(Q, K, V, mask)
        score

    def test_multihead_attention(self):
        embedding_dim = 24
        max_seq = 6
        num_heads = 3
        mha = MultiHeadAttention(embedding_dim, num_heads)
        # Rolling with self attention
        Q = torch.rand(1, max_seq, embedding_dim)
        K = Q.clone()
        V = Q.clone()
        score = mha(Q, K, V)
        score

    def test_encoder(self):
        embedding_dim = 24
        max_seq = 6
        num_heads = 8
        factor = 4
        x = torch.rand(1, max_seq, embedding_dim)
        encoder = Encoder(embedding_dim, num_heads, factor)
        z = encoder(x)
        z
    
    def test_decoder(self):
        embedding_dim = 24
        max_seq = 6
        num_heads = 8
        factor = 4
        mask = torch.triu(torch.ones(1, max_seq, max_seq), diagonal=1)
        x = torch.rand(1, max_seq, embedding_dim)
        z = torch.rand(1, max_seq, embedding_dim)
        decoder = Decoder(embedding_dim, num_heads, factor)
        y = decoder(x, z, mask)
        y
        
    def test_languagehead(self):
        vocab_size = 10
        embedding_dim = 24
        token_embedding = torch.rand(1, 1, embedding_dim)
        lh = LanguageHead(embedding_dim, vocab_size)
        logits = lh(token_embedding)
        probs = lh(token_embedding, probabilities=True)
        logits, probs

    def test_dtransformer(self):
        N = 2
        embedding_dim = 12
        vocab_size = 20
        max_seq = 6
        num_heads = 4
        factor = 4
        mask = torch.triu(torch.ones(max_seq, max_seq), diagonal=1)
        model = DTransformer(N, 
                             embedding_dim, 
                             vocab_size, 
                             max_seq, 
                             num_heads, 
                             factor, 
                             mask)
        x = torch.randint(0, 20, (1, max_seq))
        x = model(x)
        x

    def test_generate(self):
        N = 2
        embedding_dim = 12
        vocab_size = 20
        max_seq = 6
        num_heads = 4
        factor = 4
        mask = torch.triu(torch.ones(max_seq, max_seq), diagonal=1)
        model = DTransformer(N, 
                             embedding_dim, 
                             vocab_size, 
                             max_seq, 
                             num_heads, 
                             factor, 
                             mask)
        prompt = torch.randint(0, 20, (1, max_seq))
        prompt = model.generate(prompt)
        prompt

    def test_generate_char(self):
        ROOT_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        PATH = os.path.join(ROOT_, "models", "Dtransformer_200.pt")
        max_seq = 50
        de = 128
        n_heads = 4
        N = 2
        factor = 4
        data = SequenceDataset(os.path.join("data", "anna.txt"), max_seq)
        v_size = len(data._vocabulary)
        mask = torch.triu(torch.ones(max_seq, max_seq), diagonal=1)
        model = DTransformer(N, de, v_size, max_seq, n_heads, factor, mask)
        model.load_state_dict(torch.load(PATH, map_location=torch.device(device="cpu")))
        model.eval()
        prompt = data.token_encoder("Three days after the quarrel, Prince Stepan Arkadyevitch").unsqueeze(0)
        o = model.generate(prompt)
        sentence = data.token_decoder(o[0])
        sentence