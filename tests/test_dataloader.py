import unittest
import sys, os
import dvc.api
import io
import pandas as pd
sys.path.append(os.path.abspath(os.path.join('..')))

from scripts.dataloader import DataLoader

class TestDataLoader(unittest.TestCase):
     def test_read_csv(self):
        """Test the readcsv method"""
        filename = "../data/train.csv"
        dataloader = DataLoader()
        content = dvc.api.read(path=filename, repo="../", rev='v1')
        pd = dataloader.read_csv(io.StringIO(content))
        col1 = pd.columns[0]
        self.assertEqual(col1, "Store")

if __name__ == '__main__':
    unittest.main()