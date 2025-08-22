import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from config import Config

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.filtered_df = None
        self.word_encoder = None
        self.speaker_encoder = None
        
    def load_and_analyze_data(self):
        print("Loading and analyzing data...")
        
        if not self.data_path:
            raise ValueError("Data path is required. Please provide a valid CSV file path.")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if not self.data_path.endswith('.csv'):
            raise ValueError("Data file must be a CSV file.")
        
        try:
            self.df = pd.read_csv(self.data_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file: {e}")
        
        required_columns = ['file_path', 'word', 'speaker']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. Required columns: {required_columns}")
        
        print(f"Loaded {len(self.df)} samples")
        
        word_counts = self.df['word'].value_counts()
        speaker_counts = self.df['speaker'].value_counts()
        
        print(f"Found {len(word_counts)} unique words")
        print(f"Found {len(speaker_counts)} speakers")
        
        rare_words = word_counts[word_counts == 1]
        common_words = word_counts[word_counts >= Config.MIN_WORD_SAMPLES]
        
        print(f"Warning: {len(rare_words)} words have only one sample")
        print(f"Success: {len(common_words)} words have {Config.MIN_WORD_SAMPLES}+ samples")
        
        self.filtered_df = self.df[self.df['word'].isin(common_words.index)]
        print(f"Filtered data to {len(self.filtered_df)} samples from common words")
        
        self._setup_encoders()
        
    def _setup_encoders(self):
        self.word_encoder = LabelEncoder()
        self.word_encoder.fit(self.filtered_df['word'])
        
        self.speaker_encoder = LabelEncoder()
        self.speaker_encoder.fit(self.filtered_df['speaker'])
        
        print(f"Setup encoder for {len(self.word_encoder.classes_)} words")
        print(f"Setup encoder for {len(self.speaker_encoder.classes_)} speakers")
    
    def get_data_info(self):
        return {
            'total_samples': len(self.filtered_df),
            'total_words': len(self.word_encoder.classes_),
            'total_speakers': len(self.speaker_encoder.classes_),
            'average_samples_per_word': len(self.filtered_df) / len(self.word_encoder.classes_)
        }
    
    def get_filtered_data(self):
        return self.filtered_df
    
    def get_encoders(self):
        return self.word_encoder, self.speaker_encoder