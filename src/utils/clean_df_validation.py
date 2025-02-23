import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataValidator:
    """
    Class to validate the preprocessed dataset.
    """
    def __init__(self, file_path: Path):
        """
        Initializes the DataValidator class.
        :param file_path: Path to the cleaned dataset.
        """
        self.file_path = file_path
        self.df = self.load_data()
    
    def load_data(self) -> pd.DataFrame:
        """
        Loads the cleaned dataset.
        :return: Pandas DataFrame
        """
        try:
            df = pd.read_csv(self.file_path)
            logging.info(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise
    
    def check_missing_values(self) -> None:
        """
        Checks for missing values in the dataset.
        """
        missing = self.df.isnull().sum()
        logging.info(f"Missing values per column:\n{missing}")
    
    def sample_preprocessed_text(self, num_samples: int = 5) -> None:
        """
        Displays a sample of preprocessed headlines and descriptions.
        :param num_samples: Number of samples to display.
        """
        if "headline" in self.df.columns and "short_description" in self.df.columns:
            logging.info("Sample cleaned data:")
            logging.info(self.df[["headline", "short_description"]].sample(num_samples))
        else:
            logging.warning("Expected columns not found in dataset!")
    
    def validate(self) -> None:
        """
        Runs all validation checks.
        """
        logging.info("Validating cleaned dataset...")
        self.check_missing_values()
        self.sample_preprocessed_text()
        logging.info("Validation completed.")
         # Check dataset size
        logging.info(f"Dataset contains {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        
        # Display random 5 rows of the cleaned dataset
        logging.info("Displaying 5 random rows from the cleaned dataset:")
        print(self.df.sample(5))

if __name__ == "__main__":
    validator = DataValidator(Path(r"C:\Users\batti\topic-modeling\data\cleaned_news.csv"))

    validator.validate()
