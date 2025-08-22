import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder


class FixedDownSyndromeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, processor, target_length: int = 30):
        self.data = data.reset_index(drop=True)
        self.processor = processor
        self.target_length = target_length
        self.sample_rate = 16000
        
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data['text'])
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"Created updated dataset:")
        print(f"   Samples: {len(self.data)}")
        print(f"   Words: {self.num_classes}")
        print(f"   Target length: {self.target_length} seconds")
        print(f"   Words: {list(self.label_encoder.classes_)[:10]}...")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            audio, sr = librosa.load(row['audio_path'], sr=self.sample_rate)
            
            if len(audio) > 0:
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            target_samples = self.target_length * self.sample_rate
            
            if len(audio) > target_samples:
                start = (len(audio) - target_samples) // 2
                audio = audio[start:start + target_samples]
            else:
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
            
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt",
                padding="max_length",
                max_length=target_samples,
                truncation=True
            )
            
            input_features = inputs.input_features.squeeze()
            
            if input_features.shape[-1] != 3000:
                if input_features.shape[-1] < 3000:
                    padding_needed = 3000 - input_features.shape[-1]
                    input_features = torch.nn.functional.pad(
                        input_features, (0, padding_needed), mode='constant', value=0
                    )
                else:
                    input_features = input_features[..., :3000]
            
            return {
                'input_features': input_features,
                'label': torch.tensor(self.labels[idx], dtype=torch.long),
                'speaker': row['speaker'],
                'quality': row.get('quality', 'unknown'),
                'text': row['text'],
                'audio_length': len(audio) / self.sample_rate
            }
            
        except Exception as e:
            print(f"Warning: Error loading {row['audio_path']}: {e}")
            return {
                'input_features': torch.zeros(80, 3000),
                'label': torch.tensor(0, dtype=torch.long),
                'speaker': 'unknown',
                'quality': 'unknown',
                'text': 'unknown',
                'audio_length': 0.0
            }