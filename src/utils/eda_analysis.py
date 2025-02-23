import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import nltk
from nltk.corpus import stopwords
from typing import Optional
from dataloader import DataLoader  # Import DataLoader

# Paths
file_path = Path("C:/Users/batti/topic-modeling/data/Huff_news/news.jsonl")

# Ensure stopwords are downloaded
nltk.download("stopwords")


class EDAAnalysis:
    """Performs exploratory data analysis (EDA) on a dataset."""

    def __init__(self, file_path: Path):
        """
        Initializes the EDA class by loading the dataset using DataLoader.
        :param file_path: Path to the dataset file.
        """
        self.file_path = file_path
        self.loader = DataLoader(self.file_path)  # Use DataLoader
        self.df = self.loader.load_data()  # Load the dataset
        self.stop_words = set(stopwords.words("english"))

    def check_missing_values(self) -> pd.Series:
        """
        Checks for missing values in the dataset.
        :return: Series containing count of missing values per column.
        """
        missing_values = self.df.isnull().sum()
        print("Missing Values Per Column:\n", missing_values[missing_values > 0])
        return missing_values

    def drop_missing_values(self) -> None:
        """
        Drops rows where 'headline' or 'short_description' are missing.
        """
        before = len(self.df)
        self.df.dropna(subset=['headline', 'short_description'], inplace=True)
        after = len(self.df)
        print(f"Removed {before - after} rows due to missing values.")

    def text_length_distribution(self) -> None:
        """
        Plots the distribution of text lengths for headlines and short descriptions.
        """
        self.df['headline_length'] = self.df['headline'].apply(lambda x: len(x.split()))
        self.df['desc_length'] = self.df['short_description'].apply(lambda x: len(x.split()))

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(self.df['headline_length'], bins=30, kde=True, ax=ax[0])
        ax[0].set_title("Headline Length Distribution")
        ax[0].set_xlabel("Word Count")

        sns.histplot(self.df['desc_length'], bins=30, kde=True, ax=ax[1])
        ax[1].set_title("Short Description Length Distribution")
        ax[1].set_xlabel("Word Count")

        plt.tight_layout()
        plt.show()

    def compute_word_statistics(self) -> dict:
        """
        Computes basic word statistics: total words, unique words, and stopword count.
        :return: Dictionary containing statistics.
        """
        all_words = [word.lower() for text in self.df['headline'].tolist() + self.df['short_description'].tolist()
                     for word in text.split()]

        total_words = len(all_words)
        unique_words = len(set(all_words))
        stopword_count = sum(1 for word in all_words if word in self.stop_words)

        stats = {
            "total_words": total_words,
            "unique_words": unique_words,
            "stopword_count": stopword_count,
            "stopword_percentage": round((stopword_count / total_words) * 100, 2),
        }

        print("Word Statistics:", stats)
        return stats

    def plot_word_frequencies(self, remove_stopwords: bool = False) -> None:
        """
        Plots the most common words in the dataset, optionally removing stopwords.
        :param remove_stopwords: If True, removes stopwords before analysis.
        """
        all_words = [word.lower() for text in self.df['headline'].tolist() + self.df['short_description'].tolist()
                     for word in text.split()]

        if remove_stopwords:
            all_words = [word for word in all_words if word not in self.stop_words]

        word_freq = Counter(all_words).most_common(20)

        plt.figure(figsize=(12, 5))
        sns.barplot(x=[word[0] for word in word_freq], y=[word[1] for word in word_freq])
        plt.xticks(rotation=45)
        plt.title("Top 20 Most Frequent Words" + (" (Without Stopwords)" if remove_stopwords else ""))
        plt.show()

    def category_distribution(self) -> None:
        """
        Plots the distribution of categories in the dataset.
        """
        plt.figure(figsize=(12, 5))
        sns.countplot(y=self.df['category'], order=self.df['category'].value_counts().index)
        plt.title("Category Distribution")
        plt.xlabel("Count")
        plt.ylabel("Category")
        plt.show()

if __name__ == "__main__":
    # Initialize EDA
    eda = EDAAnalysis(file_path)

    # Run EDA steps
    eda.check_missing_values()
    eda.drop_missing_values()
    eda.text_length_distribution()
    eda.compute_word_statistics()
    eda.plot_word_frequencies(remove_stopwords=True)
    eda.category_distribution()