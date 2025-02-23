import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np
import spacy
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import Optional
from dataloader import DataLoader  # Import DataLoader

# Ensure required resources are available
nltk.download("stopwords")
nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")

# Paths
file_path = Path("C:/Users/batti/topic-modeling/data/Huff_news/news.jsonl")


class EDAAnalysis:
    """Performs advanced exploratory data analysis (EDA) on a news dataset."""

    def __init__(self, file_path: Path):
        """
        Initializes the EDA class by loading the dataset using DataLoader.
        :param file_path: Path to the dataset file.
        """
        self.file_path = file_path
        self.loader = DataLoader(self.file_path)  # Load dataset via DataLoader
        self.df = self.loader.load_data()
        self.stop_words = set(nltk.corpus.stopwords.words("english"))

    ## ------------------------- TEXT QUALITY ASSESSMENT -------------------------

    def check_duplicates(self) -> int:
        """Checks for duplicate headlines."""
        duplicates = self.df["headline"].duplicated().sum()
        print(f"Duplicate headlines found: {duplicates}")
        return duplicates

    def compute_lexical_diversity(self) -> float:
        """Computes lexical diversity (unique words / total words)."""
        all_words = [word.lower() for text in self.df["headline"] for word in text.split()]
        unique_words = set(all_words)
        lexical_diversity = len(unique_words) / len(all_words)
        print(f"Lexical Diversity: {lexical_diversity:.4f}")
        return lexical_diversity

    ## ------------------------- SENTIMENT & TOPIC ANALYSIS -------------------------

    def analyze_sentiment(self) -> pd.DataFrame:
        """Performs sentiment analysis on headlines using TextBlob."""
        self.df["polarity"] = self.df["headline"].apply(lambda x: TextBlob(x).sentiment.polarity)
        self.df["subjectivity"] = self.df["headline"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        return self.df[["headline", "polarity", "subjectivity"]]

    def plot_sentiment_distribution(self) -> None:
        """Plots sentiment polarity distribution."""
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df["polarity"], bins=30, kde=True)
        plt.title("Sentiment Polarity Distribution")
        plt.xlabel("Polarity")
        plt.ylabel("Frequency")
        plt.show()

    def perform_topic_modeling(self, num_topics: int = 5) -> None:
        """Performs LDA topic modeling on headlines using TF-IDF."""
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(self.df["headline"])

        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(X)

        words = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [words[i] for i in topic.argsort()[:-10 - 1:-1]]
            print(f"Topic {topic_idx + 1}: {' | '.join(top_words)}")

    ## ------------------------- NER & POS TAGGING -------------------------

    def extract_named_entities(self) -> pd.DataFrame:
        """Extracts named entities (e.g., persons, organizations)."""
        entities = []
        for text in self.df["headline"]:
            doc = nlp(text)
            for ent in doc.ents:
                entities.append((ent.text, ent.label_))
        
        entity_df = pd.DataFrame(entities, columns=["Entity", "Type"])
        return entity_df.value_counts().reset_index(name="Count")

    def analyze_pos_tags(self) -> pd.DataFrame:
        """Analyzes part-of-speech (POS) tagging distribution."""
        pos_counts = Counter()
        for text in self.df["headline"]:
            doc = nlp(text)
            pos_counts.update([token.pos_ for token in doc])
        
        return pd.DataFrame(pos_counts.items(), columns=["POS", "Count"]).sort_values(by="Count", ascending=False)

    ## ------------------------- WORD EMBEDDING VISUALIZATIONS -------------------------

    def generate_wordcloud(self) -> None:
        """Generates a word cloud of frequent words in headlines."""
        text = " ".join(self.df["headline"])
        wordcloud = WordCloud(stopwords=self.stop_words, background_color="white", width=800, height=400).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Headlines")
        plt.show()

    ## ------------------------- CATEGORY ANALYSIS -------------------------

    def compare_text_complexity(self) -> None:
        """Compares text complexity across news categories."""
        self.df["headline_length"] = self.df["headline"].apply(lambda x: len(x.split()))
        category_complexity = self.df.groupby("category")["headline_length"].mean().sort_values()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=category_complexity.index, y=category_complexity.values)
        plt.xticks(rotation=45)
        plt.xlabel("Category")
        plt.ylabel("Avg Headline Length")
        plt.title("Text Complexity by Category")
        plt.show()

if __name__ == "__main__":
    eda = EDAAnalysis(file_path)

# Text Quality
eda.check_duplicates()
eda.compute_lexical_diversity()

# Sentiment & Topic Analysis
eda.analyze_sentiment()
eda.plot_sentiment_distribution()
eda.perform_topic_modeling(num_topics=5)

# NER & POS Tagging
ner_df = eda.extract_named_entities()
pos_df = eda.analyze_pos_tags()

# Word Embeddings & Cloud
eda.generate_wordcloud()

# Category Analysis
eda.compare_text_complexity()
