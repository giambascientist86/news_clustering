import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.utils.enhanced_eda import EDAAnalysis
from src.utils.dataloader import DataLoader

class TestEDAAnalysis(unittest.TestCase):
    """Unit tests for the EDAAnalysis class."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment by creating a sample dataset."""
        cls.sample_data = [
            {
                "headline": "Breaking News: Market Crash",
                "short_description": "The stock market faced a sharp decline today.",
                "category": "Business"
            },
            {
                "headline": "New Study on Climate Change",
                "short_description": "Scientists discover alarming effects of global warming.",
                "category": "Science"
            },
            {
                "headline": "Sports Championship Results",
                "short_description": "The latest updates from the international sports championship.",
                "category": "Sports"
            }
        ]
        cls.df = pd.DataFrame(cls.sample_data)

        # Mock DataLoader to return our test DataFrame
        cls.mock_loader = MagicMock(spec=DataLoader)
        cls.mock_loader.load_data.return_value = cls.df

        # Create a temporary test file path
        cls.test_file_path = Path("test_sample.jsonl")

        # Patch DataLoader so it returns our mock data
        with patch("src.utils.enhanced_eda.DataLoader", return_value=cls.mock_loader):
            cls.eda = EDAAnalysis(cls.test_file_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        if cls.test_file_path.exists():
            cls.test_file_path.unlink()

    ## ------------------------- DATA LOADING -------------------------
    
    def test_data_loading(self):
        """Test if DataLoader correctly loads the dataset."""
        self.assertFalse(self.eda.df.empty)
        self.assertEqual(len(self.eda.df), len(self.df))

    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        with self.assertRaises(Exception):
            EDAAnalysis(Path("invalid_path.jsonl"))

    ## ------------------------- TEXT QUALITY ASSESSMENT -------------------------

    def test_check_duplicates(self):
        """Test detection of duplicate headlines."""
        self.df.loc[3] = self.df.loc[0]  # Add duplicate entry
        self.assertEqual(self.eda.check_duplicates(), 1)

    def test_lexical_diversity(self):
        """Test lexical diversity calculation."""
        diversity = self.eda.compute_lexical_diversity()
        self.assertGreater(diversity, 0)
        self.assertLessEqual(diversity, 1)

    ## ------------------------- SENTIMENT & TOPIC ANALYSIS -------------------------

    def test_analyze_sentiment(self):
        """Test sentiment analysis on headlines."""
        sentiment_df = self.eda.analyze_sentiment()
        self.assertIn("polarity", sentiment_df.columns)
        self.assertIn("subjectivity", sentiment_df.columns)
        self.assertEqual(len(sentiment_df), len(self.df))
        self.assertTrue(all(-1 <= p <= 1 for p in sentiment_df["polarity"]))

    @patch("matplotlib.pyplot.show")
    def test_plot_sentiment_distribution(self, mock_show):
        """Test that sentiment distribution plot runs without error."""
        self.eda.plot_sentiment_distribution()
        mock_show.assert_called_once()

    def test_perform_topic_modeling(self):
        """Test LDA topic modeling runs successfully."""
        with patch("builtins.print") as mock_print:
            self.eda.perform_topic_modeling(num_topics=2)
            mock_print.assert_called()

    ## ------------------------- NER & POS TAGGING -------------------------

    def test_extract_named_entities(self):
        """Test named entity extraction."""
        ner_df = self.eda.extract_named_entities()
        self.assertIn("Entity", ner_df.columns)
        self.assertIn("Type", ner_df.columns)

    def test_analyze_pos_tags(self):
        """Test POS tagging analysis."""
        pos_df = self.eda.analyze_pos_tags()
        self.assertIn("POS", pos_df.columns)
        self.assertIn("Count", pos_df.columns)
        self.assertGreater(len(pos_df), 0)

    ## ------------------------- WORD EMBEDDINGS & VISUALIZATIONS -------------------------

    @patch("matplotlib.pyplot.show")
    def test_generate_wordcloud(self, mock_show):
        """Test that word cloud visualization runs without error."""
        self.eda.generate_wordcloud()
        mock_show.assert_called_once()

    ## ------------------------- CATEGORY ANALYSIS -------------------------

    @patch("matplotlib.pyplot.show")
    def test_compare_text_complexity(self, mock_show):
        """Test that text complexity comparison plot runs without error."""
        self.eda.compare_text_complexity()
        mock_show.assert_called_once()

if __name__ == "__main__":
    unittest.main()
