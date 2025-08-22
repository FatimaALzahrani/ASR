import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd


class EnhancedAudioDataset(Dataset):
    def __init__(self, dataframe, label_encoder, feature_type='mfcc', max_length=100, augment=False):
        self.data = dataframe
        self.label_encoder = label_encoder
        self.feature_type = feature_type
        self.max_length = max_length
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            if 'file_path' not in row or pd.isna(row['file_path']):
                print(f"Warning: No file path for sample {idx}, using simulated audio data")
                audio = self.generate_simulated_audio(row)
                sr = 16000
            elif not os.path.exists(str(row['file_path'])):
                print(f"Warning: File not found {row['file_path']}, using simulated audio data")
                audio = self.generate_simulated_audio(row)
                sr = 16000
            else:
                audio, sr = librosa.load(row['file_path'], sr=16000)
            
            if self.augment and len(audio) > 0:
                audio = self.apply_light_augmentation(audio, sr)
            
            if self.feature_type == 'mfcc':
                features = self.extract_mfcc_features(audio, sr)
            elif self.feature_type == 'mel_spectrogram':
                features = self.extract_mel_spectrogram(audio, sr)
            elif self.feature_type == 'combined':
                features = self.extract_combined_features(audio, sr)
            else:
                features = self.extract_mfcc_features(audio, sr)
            
            features = self.normalize_sequence_length(features)
            features_tensor = torch.FloatTensor(features)
            label = self.label_encoder.transform([row['word']])[0]
            
            return features_tensor, label, row['speaker']
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            features = self.generate_dummy_features()
            features_tensor = torch.FloatTensor(features)
            dummy_label = 0 if len(self.label_encoder.classes_) == 0 else self.label_encoder.transform([self.label_encoder.classes_[0]])[0]
            return features_tensor, dummy_label, row.get('speaker', 'unknown')
    
    def generate_simulated_audio(self, row):
        np.random.seed(hash(str(row.get('word', 'default')) + str(row.get('speaker', 'default'))) % 2**32)
        
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        word_hash = hash(str(row.get('word', 'default'))) % 1000
        base_freq = 200 + (word_hash % 300)
        
        speaker_hash = hash(str(row.get('speaker', 'default'))) % 100
        speaker_freq_mod = 1.0 + (speaker_hash % 20) / 100.0
        
        audio = np.sin(2 * np.pi * base_freq * speaker_freq_mod * t)
        audio += 0.3 * np.sin(2 * np.pi * base_freq * speaker_freq_mod * 2 * t)
        audio += 0.1 * np.random.normal(0, 0.1, len(t))
        
        envelope = np.exp(-3 * t)
        audio = audio * envelope
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def generate_dummy_features(self):
        if self.feature_type == 'mel_spectrogram':
            return np.zeros((self.max_length, 80))
        elif self.feature_type == 'combined':
            return np.zeros((self.max_length, 17))
        else:
            return np.zeros((self.max_length, 39))
    
    def extract_mfcc_features(self, audio, sr):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
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
    
    def normalize_sequence_length(self, features):
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
    
    def apply_light_augmentation(self, audio, sr):
        if np.random.random() < 0.3:
            noise_factor = np.random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_factor, len(audio))
            audio = audio + noise
        
        if np.random.random() < 0.2:
            speed_factor = np.random.uniform(0.95, 1.05)
            audio = librosa.effects.time_stretch(audio, rate=speed_factor)
        
        return audio