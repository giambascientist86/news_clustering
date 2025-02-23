import unittest
from pathlib import Path
import pandas as pd
from src.utils.dataloader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Unit tests for DataLoader class."""
    
    def setUp(self):
        """Setup test environment."""
        self.test_file_path = Path("tests/data/sample_news.jsonl")  # Updated test path
        self.loader = DataLoader(self.test_file_path)
    
    def test_load_data(self):
        """Test that DataLoader loads data correctly."""
        df = self.loader.load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn("headline", df.columns)
        self.assertIn("category", df.columns)

if __name__ == "__main__":
    unittest.main()
