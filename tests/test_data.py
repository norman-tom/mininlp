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
        tokenizer1.save(os.path.join("models", "voc_test.pkl"))
        tokenizer2 = Tokenizer(pickle.load(open(os.path.join("models", "voc_test.pkl"), "rb")))
        assert tokenizer2._vocabulary == tokenizer1._vocabulary
        tokenizer2