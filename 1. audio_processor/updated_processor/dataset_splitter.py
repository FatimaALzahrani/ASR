import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict


class DatasetSplitter:
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def create_dataset_splits(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        speakers = data['speaker'].unique()
        speakers = [s for s in speakers if s != 'Unknown']
        
        if len(speakers) < 3:
            print("Warning: Few speakers available, splitting by files")
            train_data, test_data = train_test_split(
                data, test_size=self.test_size, random_state=self.random_state, 
                stratify=data['word']
            )
            train_data, val_data = train_test_split(
                train_data, test_size=self.val_size/(1-self.test_size), 
                random_state=self.random_state, stratify=train_data['word']
            )
        else:
            train_speakers, test_speakers = train_test_split(
                speakers, test_size=self.test_size, random_state=self.random_state
            )
            
            if len(train_speakers) > 1:
                train_speakers, val_speakers = train_test_split(
                    train_speakers, test_size=self.val_size/(1-self.test_size), 
                    random_state=self.random_state
                )
            else:
                val_speakers = train_speakers[:1]
            
            train_data = data[data['speaker'].isin(train_speakers)]
            val_data = data[data['speaker'].isin(val_speakers)]
            test_data = data[data['speaker'].isin(test_speakers)]
        
        print(f"\nDataset split:")
        print(f"   Training: {len(train_data)} samples")
        print(f"   Validation: {len(val_data)} samples")
        print(f"   Test: {len(test_data)} samples")
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }