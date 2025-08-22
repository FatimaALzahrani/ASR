import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


class FixedAudioDataset(Dataset):
    
    def __init__(self, dataframe, feature_type='mfcc', max_length=100, augment=False):
        self.data = dataframe
        self.feature_type = feature_type
        self.max_length = max_length
        self.augment = augment
        self.search_paths = ['data/enhanced', 'data/clean', '.']
        
    def __len__(self):
        return len(self.data)
    
    def find_audio_file(self, file_path, word):
        filename = Path(file_path).name
        basename = filename.replace('_processed.wav', '.wav').replace('enhanced_', '')
        
        search_patterns = [
            file_path,
            f"data/enhanced/{word}/enhanced_{basename}",
            f"data/clean/{word}/{basename}",
            f"data/clean/{word}/{filename}",
        ]
        
        try:
            file_number = ''.join(filter(str.isdigit, basename.split('.')[0]))
            if file_number:
                search_patterns.extend([
                    f"data/clean/{word}/{file_number}.wav",
                    f"data/clean/{word}/0{file_number}.wav" if len(file_number) < 2 else f"data/clean/{word}/{file_number}.wav",
                ])
        except:
            pass
        
        for pattern in search_patterns:
            if Path(pattern).exists():
                return pattern
        
        word_dir = Path(f"data/clean/{word}")
        if word_dir.exists():
            for wav_file in word_dir.glob("*.wav"):
                return str(wav_file)
        
        return None
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            audio_path = self.find_audio_file(row['file_path'], row['word'])
            
            if audio_path is None:
                print(f"Warning: File not found: {row['filename']}")
                return self.get_empty_features(), 0, row.get('speaker', 'unknown')
            
            audio, sr = librosa.load(audio_path, sr=16000)
            
            if self.augment and np.random.random() < 0.15:
                audio = self.light_augmentation(audio)
            
            if self.feature_type == 'mfcc':
                features = self.extract_mfcc_features(audio, sr)
            elif self.feature_type == 'mel_spectrogram':
                features = self.extract_mel_spectrogram(audio, sr)
            elif self.feature_type == 'combined':
                features = self.extract_combined_features(audio, sr)
            else:
                features = self.extract_mfcc_features(audio, sr)
            
            features = self.normalize_length(features)
            features_tensor = torch.FloatTensor(features)
            
            return features_tensor, row['word'], row.get('speaker', 'unknown')
            
        except Exception as e:
            print(f"Error processing {row.get('filename', 'unknown')}: {e}")
            return self.get_empty_features(), row['word'], row.get('speaker', 'unknown')
    
    def get_empty_features(self):
        if self.feature_type == 'mel_spectrogram':
            return torch.zeros(self.max_length, 80)
        elif self.feature_type == 'combined':
            return torch.zeros(self.max_length, 17)
        else:
            return torch.zeros(self.max_length, 39)
    
    def extract_mfcc_features(self, audio, sr):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        features = np.vstack([mfccs, delta, delta2])
        return features.T
    
    def extract_mel_spectrogram(self, audio, sr):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, n_fft=512, hop_length=160)
        log_mel_spec = librosa.power_to_db(mel_spec)
        return log_mel_spec.T
    
    def extract_combined_features(self, audio, sr):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features = np.vstack([mfccs, spectral_centroids, spectral_bandwidth, spectral_rolloff, zcr])
        return features.T
    
    def normalize_length(self, features):
        if len(features) > self.max_length:
            start = (len(features) - self.max_length) // 2
            features = features[start:start + self.max_length]
        elif len(features) < self.max_length:
            padding = self.max_length - len(features)
            if len(features.shape) == 1:
                features = np.pad(features, (0, padding), mode='constant')
            else:
                features = np.pad(features, ((0, padding), (0, 0)), mode='constant')
        return features
    
    def light_augmentation(self, audio):
        if np.random.random() < 0.3:
            noise_factor = np.random.uniform(0.001, 0.003)
            noise = np.random.normal(0, noise_factor, len(audio))
            audio = audio + noise
        return audio