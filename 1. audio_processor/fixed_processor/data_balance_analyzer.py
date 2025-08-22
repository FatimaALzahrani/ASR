import pandas as pd
from typing import Dict


class DataBalanceAnalyzer:
    def __init__(self, numpy_converter):
        self.numpy_converter = numpy_converter
    
    def analyze_data_balance(self, df: pd.DataFrame) -> Dict:
        print("\nAnalyzing data balance...")
        
        word_counts = df['word'].value_counts()
        speaker_counts = df['speaker'].value_counts()
        quality_counts = df['speech_quality'].value_counts()
        
        single_sample_words = word_counts[word_counts == 1].index.tolist()
        few_sample_words = word_counts[(word_counts >= 2) & (word_counts < 5)].index.tolist()
        moderate_sample_words = word_counts[(word_counts >= 5) & (word_counts < 15)].index.tolist()
        many_sample_words = word_counts[word_counts >= 15].index.tolist()
        
        balance_info = {
            'word_distribution': self.numpy_converter.convert_numpy_types(word_counts.to_dict()),
            'speaker_distribution': self.numpy_converter.convert_numpy_types(speaker_counts.to_dict()),
            'quality_distribution': self.numpy_converter.convert_numpy_types(quality_counts.to_dict()),
            'single_sample_words': single_sample_words,
            'few_sample_words': few_sample_words,
            'moderate_sample_words': moderate_sample_words,
            'many_sample_words': many_sample_words,
            'total_words': int(len(word_counts)),
            'total_samples': int(len(df)),
            'avg_samples_per_word': float(word_counts.mean()),
            'min_samples_per_word': int(word_counts.min()),
            'max_samples_per_word': int(word_counts.max())
        }
        
        print(f"Total words: {balance_info['total_words']}")
        print(f"Total samples: {balance_info['total_samples']}")
        print(f"Average samples per word: {balance_info['avg_samples_per_word']:.1f}")
        print(f"Words with single sample: {len(single_sample_words)}")
        print(f"Words with few samples (2-4): {len(few_sample_words)}")
        print(f"Words with moderate samples (5-14): {len(moderate_sample_words)}")
        print(f"Words with many samples (15+): {len(many_sample_words)}")
        
        return balance_info