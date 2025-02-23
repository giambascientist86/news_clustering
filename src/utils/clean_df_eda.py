import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import argparse
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_cleaned_data(file_path: Path) -> pd.DataFrame:
    """Loads the cleaned dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Cleaned dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def display_random_samples(df: pd.DataFrame, num_samples: int = 5) -> None:
    """Displays random samples from the cleaned dataset."""
    logging.info(f"Displaying {num_samples} random samples from the dataset.")
    print(df.sample(num_samples))

def plot_headline_length_distribution(df: pd.DataFrame):
    """Plots the distribution of headline lengths."""
    if "cleaned_headline" not in df.columns:
        raise KeyError("Column 'cleaned_headline' is missing. Ensure preprocessing is applied before running EDA.")

    df["headline_length"] = df["cleaned_headline"].apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(10, 5))
    sns.histplot(df["headline_length"], bins=20, kde=True)
    plt.xlabel("Headline Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Cleaned Headline Lengths")
    plt.show()


def generate_wordcloud(df: pd.DataFrame) -> None:
    """Generates a word cloud from the cleaned headlines."""
    text = " ".join(df["cleaned_headline"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Cleaned Headlines")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Perform EDA on cleaned dataset.")
    parser.add_argument("--file", type=str, required=True, help="Path to the cleaned dataset CSV file.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"File {args.file} not found. Check the path and try again.")

    df = pd.read_csv(args.file)

    print("\nDataset Preview:")
    print(df.head())

    # Ensure 'cleaned_headline' exists before visualization
    if "cleaned_headline" not in df.columns:
        print("Error: 'cleaned_headline' column missing from dataset. Check preprocessing step.")
        return

    plot_headline_length_distribution(df)

if __name__ == "__main__":
    main()