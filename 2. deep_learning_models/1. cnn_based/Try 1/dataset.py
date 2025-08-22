import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from audio_processor import AudioProcessor


class SpeechDataset(Dataset):
    
    def __init__(self, csv_path, mappings, enhanced_dir="C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Codes/Clean/1. audio_processor/down_syndrome_processor/data/enhanced", 
                 audio_processor=None, augment=False):
        self.df = pd.read_csv(csv_path, encoding='utf-8')
        self.word_to_id = mappings['word_to_id']
        self.enhanced_dir = enhanced_dir
        self.augment = augment
        
        if audio_processor is None:
            self.audio_processor = AudioProcessor()
        else:
            self.audio_processor = audio_processor
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Create enhanced file name
        enhanced_name = row['word'] + f"/enhanced_{row['filename']}"
        
        # Load and process audio
        audio = self.audio_processor.load_audio(
            row['file_path'], 
            self.enhanced_dir, 
            enhanced_name
        )
        
        # Apply augmentation if training
        if self.augment:
            audio = self.audio_processor.augment_audio(audio)
        
        # Normalize length and extract features
        audio = self.audio_processor.normalize_length(audio, self.augment)
        features = self.audio_processor.extract_features(audio)
        
        # Get label
        label = self.word_to_id[row['word']]
        
        return features, label, row['speaker']


def collate_fn(batch):
    features, labels, speakers = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)
    return features_padded, torch.tensor(labels), speakers