import unittest
from mininlp.data import SequenceDataset
from mininlp.data import Tokenizer
import os
import pickle

class TestData(unittest.TestCase):
    def test_sequence(self):
        data = SequenceDataset(os.path.join("data", "anna.txt"), 20)
        data

    def test_tokenizer(self):
        raw = open(os.path.join("data", "anna.txt")).read()
        voc = set(raw)
        tokenizer1 = Tokenizer(voc)
        tokenizer1.save(os.path.join("models", "vocab_test.pkl"))
        tokenizer2 = Tokenizer()
        tokenizer2.load(os.path.join("models", "vocab_test.pkl"))

    def test_token_load(self):
        tokenizer = Tokenizer()
        tokenizer.load(os.path.join("models", "vocab_test.pkl"))
        tokenizer