"""
Advanced Data Processing Pipeline for Book Metadata

This module provides a robust data cleaning and feature engineering pipeline with:
- Comprehensive data validation
- Text normalization using NLP techniques
- Metadata tracking and reporting
- Error handling and logging
- Feature engineering for downstream analytics
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
import json
import logging
from pathlib import Path

# # Initialize NLP resources
nltk.download('punkt')
stemmer = LancasterStemmer()

class DataCleaner:
    """
    A comprehensive data cleaning and feature engineering pipeline for book metadata.
    """
    def __init__(self):
        self.input_path = r"C:\Users\Sha'Monica\Desktop\ITEC 5020\book data\processed\book1-100k_cleaned.csv"
        self.df = None
        self.metadata = {
            'original_count': 0,
            'cleaned_count': 0,
            'invalid_ratings_removed': 0,
            'invalid_years_removed': 0,
            'empty_authors_removed': 0,
            'duplicates_removed': 0,
            'vocab_size': 0
        }
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _validate_data(self):
        """Enhanced data quality validation"""
        if self.df is None:
            return self

        # Column renaming and validation
        self.df = self.df.rename(columns={
            'Authors': 'authors',
            'Rating': 'rating',
            'PublishYear': 'publish_year',
            'Name': 'title',
            'pagesNumber': 'pages_number',
            'CountsOfReview': 'counts_of_review'
        })

        # Check required columns
        required_cols = ['book_id', 'title', 'authors', 'rating', 'publish_year']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        # Type conversion
        self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')
        self.df['publish_year'] = pd.to_numeric(self.df['publish_year'], errors='coerce')

        # Data validation
        valid_ratings = self.df['rating'].between(0, 5)
        valid_years = self.df['publish_year'].between(1800, 2025)
        valid_authors = self.df['authors'].str.strip() != ''
        
        self.metadata['invalid_ratings_removed'] = len(self.df[~valid_ratings])
        self.metadata['invalid_years_removed'] = len(self.df[~valid_years])
        self.metadata['empty_authors_removed'] = len(self.df[~valid_authors])
        
        self.df = self.df[valid_ratings & valid_years & valid_authors]
        return self

    # --------------------------
    # Core Processing Methods
    # --------------------------
    def _load_data(self):
        """Load data with error handling"""
        try:
            self.df = pd.read_csv(self.input_path)
            self.metadata['original_count'] = len(self.df)
            logging.info(f"Successfully loaded {len(self.df)} records")
            return True
        except Exception as e:
            logging.error(f"Loading failed: {str(e)}")
            return False

    def _remove_duplicates(self):
        """Advanced duplicate removal"""
        if self.df is None:
            return self
            
        initial_count = len(self.df)
        if 'book_id' in self.df.columns:
            self.df = self.df.drop_duplicates(subset='book_id', keep='first')
            self.metadata['duplicates_removed'] = initial_count - len(self.df)
            logging.info(f"Removed {self.metadata['duplicates_removed']} duplicates")
        else:
            logging.warning("'book_id' column not found for duplicate removal")
        return self

    def _clean_text(self, text):
        """Multilingual text normalization pipeline"""
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        tokens = word_tokenize(text)
        return ' '.join([stemmer.stem(token) for token in tokens])

    def _enhance_features(self):
        """Feature engineering with temporal indexing"""
        if self.df is None:
            return self
            
        # Clean title
        if 'title' in self.df.columns:
            self.df['clean_title'] = self.df['title'].apply(self._clean_text)
            self.metadata['vocab_size'] = len(set(' '.join(self.df['clean_title']).split()))
        
        # Author tokens
        if 'authors' in self.df.columns:
            self.df['author_tokens'] = self.df['authors'].apply(
                lambda x: json.dumps(list(set(self._clean_text(x).split())))
            )
        
        # Publication decade
        if 'publish_year' in self.df.columns:
            self.df['pub_decade'] = (self.df['publish_year'] // 10) * 10
        return self

    # --------------------------
    # Main Processing Pipeline
    # --------------------------
    def process(self):  # <-- Properly indented under the class
        """End-to-end processing pipeline"""
        output_path = r"C:\Users\Sha'Monica\Desktop\Coursework\chatbot\processed_books.parquet"
        
        if not self._load_data():
            raise ValueError("Data loading failed")
        
        try:
            (self._validate_data()
               ._remove_duplicates()
               ._enhance_features())
            
            self.metadata['cleaned_count'] = len(self.df)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self.df.to_parquet(output_path)
            with open(Path(output_path).parent / 'metadata.json', 'w') as f:
                json.dump(self.metadata, f)
            
            logging.info(f"✅ Processing complete. Saved to {output_path}")
            return self.df
            
        except Exception as e:
            logging.error(f"🚨 Processing failed: {str(e)}")
            raise

# ==============================
# Execution Block
# ==============================
if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaned_data = cleaner.process()
