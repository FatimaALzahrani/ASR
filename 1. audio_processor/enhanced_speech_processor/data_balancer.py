import pandas as pd
from typing import Dict


class DataBalancer:
    
    def __init__(self):
        self.balance_info = {}
    
    def analyze_data_balance(self, df: pd.DataFrame) -> Dict:
        print("Analyzing data balance...")
        
        # Word distribution
        word_counts = df['word'].value_counts()
        
        # Speaker distribution
        speaker_counts = df['speaker'].value_counts()
        
        # Quality distribution
        quality_counts = df['speech_quality'].value_counts()
        
        # Rare words (less than 3 samples)
        rare_words = word_counts[word_counts < 3].index.tolist()
        
        # Common words (more than 20 samples)
        common_words = word_counts[word_counts > 20].index.tolist()
        
        self.balance_info = {
            'word_distribution': word_counts.to_dict(),
            'speaker_distribution': speaker_counts.to_dict(),
            'quality_distribution': quality_counts.to_dict(),
            'rare_words': rare_words,
            'common_words': common_words,
            'total_words': len(word_counts),
            'total_samples': len(df),
            'avg_samples_per_word': word_counts.mean(),
            'min_samples_per_word': word_counts.min(),
            'max_samples_per_word': word_counts.max()
        }
        
        self._print_balance_summary()
        
        return self.balance_info
    
    def _print_balance_summary(self):
        print(f"Total words: {self.balance_info['total_words']}")
        print(f"Total samples: {self.balance_info['total_samples']}")
        print(f"Average samples per word: {self.balance_info['avg_samples_per_word']:.1f}")
        print(f"Rare words (< 3 samples): {len(self.balance_info['rare_words'])}")
        print(f"Common words (> 20 samples): {len(self.balance_info['common_words'])}")
    
    def get_word_statistics(self) -> Dict:
        if not self.balance_info:
            return {}
        
        return {
            'word_counts': self.balance_info['word_distribution'],
            'rare_words': self.balance_info['rare_words'],
            'common_words': self.balance_info['common_words']
        }
    
    def get_speaker_statistics(self) -> Dict:
        if not self.balance_info:
            return {}
        
        return {
            'speaker_counts': self.balance_info['speaker_distribution'],
            'quality_distribution': self.balance_info['quality_distribution']
        }
    
    def filter_rare_words(self, df: pd.DataFrame, min_samples: int = 2) -> pd.DataFrame:
        word_counts = df['word'].value_counts()
        valid_words = word_counts[word_counts >= min_samples].index
        filtered_df = df[df['word'].isin(valid_words)].copy()
        
        excluded_count = len(df) - len(filtered_df)
        if excluded_count > 0:
            print(f"Excluded {excluded_count} samples from rare words")
        
        return filtered_df