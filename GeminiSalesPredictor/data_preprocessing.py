import pandas as pd
import numpy as np
from datetime import datetime
import re

class DataPreprocessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.processed_df = None
        
    def load_data(self):
        """Load the PlayStation sales dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def clean_data(self):
        """Clean and preprocess the data"""
        print("\nCleaning data...")
        df = self.df.copy()
        
        # Handle missing values
        df['rating'] = df['rating'].fillna(df['rating'].median())
        df['ratings_count'] = df['ratings_count'].fillna(0)
        df['metacritic'] = df['metacritic'].fillna(df['metacritic'].median())
        df['genres'] = df['genres'].fillna('Unknown')
        df['Total Shipped'] = df['Total Shipped'].fillna(0)
        
        # Fill missing Last Update with Release Date
        df['Last Update'] = df['Last Update'].fillna(df['Release Date'])
        
        # Convert dates to datetime
        df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
        df['Last Update'] = pd.to_datetime(df['Last Update'], errors='coerce')
        
        # Extract year and month from release date
        df['Release Year'] = df['Release Date'].dt.year
        df['Release Month'] = df['Release Date'].dt.month
        df['Release Year'] = df['Release Year'].fillna(df['Release Year'].median())
        df['Release Month'] = df['Release Month'].fillna(6)  # Default to mid-year
        
        # Remove rows with missing critical values
        df = df.dropna(subset=['Total Sales'])
        
        print(f"Cleaned dataset: {df.shape[0]} rows")
        self.df = df
        return df
    
    def engineer_features(self):
        """Create engineered features for ML models"""
        print("\nEngineering features...")
        df = self.df.copy()
        
        # Console encoding (PS3, PS4, PS5)
        df['Console_PS3'] = (df['Console'] == 'PS3').astype(int)
        df['Console_PS4'] = (df['Console'] == 'PS4').astype(int)
        df['Console_PS5'] = (df['Console'] == 'PS5').astype(int)
        
        # Genre features - count number of genres
        df['Genre_Count'] = df['genres'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        
        # Platform count - number of platforms the game is on
        df['Platform_Count'] = df['platforms'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        
        # Sales ratios
        df['NA_Sales_Ratio'] = df['NA Sales'] / (df['Total Sales'] + 1)
        df['PAL_Sales_Ratio'] = df['PAL Sales'] / (df['Total Sales'] + 1)
        df['Japan_Sales_Ratio'] = df['Japan Sales'] / (df['Total Sales'] + 1)
        df['Other_Sales_Ratio'] = df['Other Sales'] / (df['Total Sales'] + 1)
        
        # Publisher popularity (games count)
        publisher_counts = df['Publisher'].value_counts()
        df['Publisher_Popularity'] = df['Publisher'].map(publisher_counts)
        
        # Developer popularity (games count)
        developer_counts = df['Developer'].value_counts()
        df['Developer_Popularity'] = df['Developer'].map(developer_counts)
        
        # Text features for transformer
        df['Text_Features'] = df.apply(
            lambda row: f"{row['Name']} | {row['Publisher']} | {row['Developer']} | {row['genres']} | {row['Console']}",
            axis=1
        )
        
        print(f"Features engineered: {df.shape[1]} total columns")
        self.processed_df = df
        return df
    
    def get_ml_features(self):
        """Get features ready for traditional ML models (XGBoost, Random Forest)"""
        feature_columns = [
            'Total Shipped', 'NA Sales', 'PAL Sales', 'Japan Sales', 'Other Sales',
            'rating', 'ratings_count', 'metacritic',
            'Release Year', 'Release Month',
            'Console_PS3', 'Console_PS4', 'Console_PS5',
            'Genre_Count', 'Platform_Count',
            'Publisher_Popularity', 'Developer_Popularity'
        ]
        
        df = self.processed_df[feature_columns].copy()
        df = df.fillna(0)
        return df
    
    def get_target(self):
        """Get target variable (Total Sales)"""
        return self.processed_df['Total Sales'].values
    
    def get_text_features(self):
        """Get text features for transformer models"""
        return self.processed_df['Text_Features'].values
    
    def get_full_dataframe(self):
        """Get the full processed dataframe"""
        return self.processed_df
    
    def process_all(self):
        """Run full preprocessing pipeline"""
        self.load_data()
        self.clean_data()
        self.engineer_features()
        print("\n✓ Data preprocessing complete!")
        return self.processed_df


if __name__ == "__main__":
    # Test the preprocessing
    preprocessor = DataPreprocessor("attached_assets/PlayStation Sales and Metadata (PS3PS4PS5) (Oct 2025)_1762234789301.csv")
    df = preprocessor.process_all()
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"\nSample features:\n{df.head()}")
