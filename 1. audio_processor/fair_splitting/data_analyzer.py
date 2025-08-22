import pandas as pd
from pathlib import Path


class DataAnalyzer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
    
    def load_current_data(self):
        print("Loading current data...")
        
        try:
            self.train_df = pd.read_csv(self.data_path / "train.csv", encoding='utf-8')
            self.val_df = pd.read_csv(self.data_path / "validation.csv", encoding='utf-8')
            self.test_df = pd.read_csv(self.data_path / "test.csv", encoding='utf-8')
            
            self.all_data = pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
            
            print(f"Total data loaded: {len(self.all_data)} samples")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def analyze_current_distribution(self):
        print("\nAnalyzing current problematic distribution:")
        print("=" * 50)
        
        test_speaker_counts = self.test_df['speaker'].value_counts()
        print("Speaker distribution in current test set:")
        for speaker, count in test_speaker_counts.items():
            percentage = (count / len(self.test_df)) * 100
            print(f"  {speaker}: {count} samples ({percentage:.1f}%)")
        
        print(f"\nStatistics per speaker:")
        for speaker in self.all_data['speaker'].unique():
            speaker_data = self.all_data[self.all_data['speaker'] == speaker]
            unique_words = len(speaker_data['word'].unique())
            total_samples = len(speaker_data)
            print(f"  {speaker}: {total_samples} samples, {unique_words} unique words")
    
    def analyze_new_distribution(self, train_df, val_df, test_df):
        print(f"\nNew fair distribution:")
        print("=" * 50)
        
        print(f"Dataset sizes:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples") 
        print(f"  Test: {len(test_df)} samples")
        
        print(f"\nSpeaker distribution in new test set:")
        if not test_df.empty:
            test_speaker_counts = test_df['speaker'].value_counts()
            for speaker, count in test_speaker_counts.items():
                percentage = (count / len(test_df)) * 100
                print(f"  {speaker}: {count} samples ({percentage:.1f}%)")
        
        print(f"\nDistribution comparison:")
        print("Speaker  | Train  | Val   | Test   | Total")
        print("-" * 50)
        
        for speaker in self.all_data['speaker'].unique():
            train_count = len(train_df[train_df['speaker'] == speaker]) if not train_df.empty else 0
            val_count = len(val_df[val_df['speaker'] == speaker]) if not val_df.empty else 0
            test_count = len(test_df[test_df['speaker'] == speaker]) if not test_df.empty else 0
            total = train_count + val_count + test_count
            
            print(f"{speaker:8} | {train_count:6} | {val_count:5} | {test_count:6} | {total:7}")