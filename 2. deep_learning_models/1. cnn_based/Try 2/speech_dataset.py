import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


class SpeechDataset(Dataset):
    def __init__(self, csv_path, mappings, enhanced_dir="data/enhanced", target_sr=16000):
        self.df = pd.read_csv(csv_path, encoding='utf-8')
        self.word_to_id = mappings['word_to_id']
        self.enhanced_dir = Path(enhanced_dir)
        self.target_sr = target_sr
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load enhanced audio if available
        enhanced_file = self.enhanced_dir / row['word'] / f"enhanced_{row['filename']}"
        if enhanced_file.exists():
            audio, sr = torchaudio.load(enhanced_file)
        else:
            audio, sr = torchaudio.load(row['file_path'])
        
        # Ensure mono audio
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        
        # Extract MFCC features
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.target_sr, 
            n_mfcc=13, 
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 64}
        )
        mfcc = mfcc_transform(audio)  # Shape: (channels, n_mfcc, time)
        
        mfcc = mfcc.squeeze(0).permute(1, 0)  # Shape: (time, n_mfcc)
        
        label = self.word_to_id[row['word']]
        return mfcc, label


def collate_fn(batch):
    mfccs, labels = zip(*batch)
    mfccs_padded = pad_sequence([mfcc for mfcc in mfccs], batch_first=True)
    return mfccs_padded, torch.tensor(labels)