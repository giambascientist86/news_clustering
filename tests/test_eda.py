import unittest
from pathlib import Path
from src.utils.eda_analysis import EDAAnalysis

class TestEDAAnalysis(unittest.TestCase):
    """Unit tests for EDAAnalysis class."""

    def setUp(self):
        """Initialize EDA with a test dataset file."""
        self.file_path = Path("data/Huff_news/sample_test_dataset.jsonl")  # Sample dataset
        self.eda = EDAAnalysis(self.file_path)

    def test_missing_values(self):
        """Test missing values detection."""
        missing_values = self.eda.check_missing_values()
        self.assertTrue(missing_values.sum() >= 0)  # Should return a valid count

    def test_text_length(self):
        """Test text length computation."""
        self.eda.text_length_distribution()
        self.assertTrue("headline_length" in self.eda.df.columns)

    def test_word_statistics(self):
        """Test word statistics computation."""
        stats = self.eda.compute_word_statistics()
        self.assertTrue(stats["total_words"] > 0)

if __name__ == "__main__":
    unittest.main()
