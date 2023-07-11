import unittest
from mininlp.data import SequenceDataset
import os

class TestData(unittest.TestCase):
    def test_sequence(self):
        data = SequenceDataset(os.path.join("data", "anna.txt"), 20)
        data