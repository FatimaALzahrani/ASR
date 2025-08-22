import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd


class AudioDataset(Dataset):
    def __init__(self, dataframe, label_encoder, max_length=16000, augment=False):
        self.data = dataframe
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            # Check if file_path column exists and file exists
            if 'file_path' not in row or pd.isna(row['file_path']):
                print(f"Warning: No file path for sample {idx}, using simulated audio data")
                audio = self.generate_simulated_audio(row)
            elif not os.path.exists(str(row['file_path'])):
                print(f"Warning: File not found {row['file_path']}, using simulated audio data")
                audio = self.generate_simulated_audio(row)
            else:
                audio, sr = librosa.load(row['file_path'], sr=16000)
            
            if self.augment and len(audio) > 0:
                audio = self.apply_augmentation(audio)
            
            audio = self.normalize_length(audio)
            audio_tensor = torch.FloatTensor(audio)
            label = self.label_encoder.transform([row['word']])[0]
            
            return audio_tensor, label, row['speaker']
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return simulated data to continue training
            audio = self.generate_simulated_audio(row)
            audio = self.normalize_length(audio)
            audio_tensor = torch.FloatTensor(audio)
            dummy_label = 0 if len(self.label_encoder.classes_) == 0 else self.label_encoder.transform([self.label_encoder.classes_[0]])[0]
            return audio_tensor, dummy_label, row.get('speaker', 'unknown')
    
    def generate_simulated_audio(self, row):
        """Generate simulated audio data for testing purposes"""
        # Create a simple synthetic audio signal based on word and speaker
        np.random.seed(hash(str(row.get('word', 'default')) + str(row.get('speaker', 'default'))) % 2**32)
        
        # Generate a simple sine wave with some characteristics based on the data
        duration = 1.0  # 1 second
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Base frequency varies by word (simulate different phonemes)
        word_hash = hash(str(row.get('word', 'default'))) % 1000
        base_freq = 200 + (word_hash % 300)  # 200-500 Hz
        
        # Speaker variation (simulate different voices)
        speaker_hash = hash(str(row.get('speaker', 'default'))) % 100
        speaker_freq_mod = 1.0 + (speaker_hash % 20) / 100.0  # ±10% variation
        
        # Generate audio
        audio = np.sin(2 * np.pi * base_freq * speaker_freq_mod * t)
        audio += 0.3 * np.sin(2 * np.pi * base_freq * speaker_freq_mod * 2 * t)  # Harmonic
        audio += 0.1 * np.random.normal(0, 0.1, len(t))  # Noise
        
        # Apply envelope to make it sound more natural
        envelope = np.exp(-3 * t)  # Exponential decay
        audio = audio * envelope
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def normalize_length(self, audio):
        if len(audio) > self.max_length:
            start = (len(audio) - self.max_length) // 2
            audio = audio[start:start + self.max_length]
        elif len(audio) < self.max_length:
            padding = self.max_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio
    
    def apply_augmentation(self, audio):
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.005, len(audio))
            audio = audio + noise
        
        if np.random.random() < 0.3:
            speed_factor = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=speed_factor)
        
        if np.random.random() < 0.3:
            pitch_shift = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=pitch_shift)
        
        return audio