import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

from scripts.dataloader import DataLoader

class TestDataLoader(unittest.TestCase):
     def test_read_csv(self):
        """Test the readcsv method"""
        filename = "../data/train.csv"
        dataloader = DataLoader()
        pd = dataloader.read_csv(filename)
        col1 = pd.columns[0]
        self.assertEqual(col1, "Store")

if __name__ == '__main__':
    unittest.main()
