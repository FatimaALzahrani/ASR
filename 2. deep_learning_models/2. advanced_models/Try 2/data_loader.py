import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    
    def __init__(self):
        self.all_data = None
        self.label_encoder = None
        self.word_to_id = None
        
    def load_all_data(self):
        print("Loading all data for maximum usage...")
        
        train_df = pd.read_csv("data/processed/train.csv", encoding='utf-8')
        val_df = pd.read_csv("data/processed/validation.csv", encoding='utf-8')
        test_df = pd.read_csv("data/processed/test.csv", encoding='utf-8')
        
        self.all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        print(f"Total data: {len(self.all_data)} samples")
        print(f"Instead of: {len(train_df)} training only")
        print(f"Data increase: +{len(val_df) + len(test_df)} samples ({((len(val_df) + len(test_df))/len(train_df))*100:.1f}%)")
        
        print(f"\nComplete data distribution:")
        for speaker in self.all_data['speaker'].unique():
            speaker_count = len(self.all_data[self.all_data['speaker'] == speaker])
            speaker_words = len(self.all_data[self.all_data['speaker'] == speaker]['word'].unique())
            print(f"  {speaker}: {speaker_count} samples, {speaker_words} words")
    
    def setup_encoders(self):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.all_data['word'])
        
        self.word_to_id = {word: idx for idx, word in enumerate(self.label_encoder.classes_)}
        print(f"Encoder setup for {len(self.label_encoder.classes_)} words")
    
    def get_data(self):
        return self.all_data
    
    def get_encoders(self):
        return self.label_encoder, self.word_to_id