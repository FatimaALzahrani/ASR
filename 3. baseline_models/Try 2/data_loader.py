import os
import json
import pandas as pd
from pathlib import Path
from config import Config

class DataLoader:
    def __init__(self):
        self.data_dir = Path(Config.DATA_DIR)
        self.test_data = None
        self.sample_data = None
        self.mappings = None
        
    def load_data(self):
        print("Loading data...")
        
        self.test_data = pd.read_csv(self.data_dir / "test.csv")
        
        train_data = pd.read_csv(self.data_dir / "train.csv")
        self.sample_data = train_data.sample(
            n=min(Config.SAMPLE_SIZE, len(train_data)), 
            random_state=Config.RANDOM_STATE
        )
        
        print(f"   Test data: {len(self.test_data)} samples")
        print(f"   Sample data: {len(self.sample_data)} samples")
        
        with open(self.data_dir / "mappings.json", 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)
    
    def get_test_data(self):
        return self.test_data
    
    def get_sample_data(self):
        return self.sample_data
    
    def get_mappings(self):
        return self.mappings
    
    def get_data_info(self):
        return {
            "test_samples": len(self.test_data),
            "sample_samples": len(self.sample_data),
            "total_words": self.mappings["num_words"],
            "total_speakers": self.mappings["num_speakers"]
        }