import json
import random
from pathlib import Path

# Define the paths
full_dataset_path = Path("C:/Users/batti/topic-modeling/data/Huff_news/news.jsonl")
sample_dataset_path = Path("data/sample_news.jsonl")

# Ensure the test data folder exists
sample_dataset_path.parent.mkdir(parents=True, exist_ok=True)

# Read the full dataset
with open(full_dataset_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Extract a small random sample (e.g., 10 articles)
sample_size = 10
sample_lines = random.sample(lines, min(sample_size, len(lines)))  # Ensure we don't sample more than available

# Save the sample dataset
with open(sample_dataset_path, "w", encoding="utf-8") as f:
    f.writelines(sample_lines)

print(f"Sample dataset created at: {sample_dataset_path}")
