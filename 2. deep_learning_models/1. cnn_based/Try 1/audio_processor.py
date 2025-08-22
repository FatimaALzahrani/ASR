import torch
import torchaudio
import torchaudio.transforms as T
import random
from pathlib import Path


class AudioProcessor:
    
    def __init__(self, target_sr=16000, target_length_sec=3.0):
        self.target_sr = target_sr
        self.target_length = int(target_length_sec * target_sr)
        
        self.mfcc_transform = T.MFCC(
            sample_rate=target_sr, 
            n_mfcc=13, 
            melkwargs={
                'n_fft': 512, 
                'hop_length': 160, 
                'n_mels': 40, 
                'f_min': 80, 
                'f_max': 8000
            }
        )
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr, 
            n_fft=512, 
            hop_length=160, 
            n_mels=40, 
            f_min=80, 
            f_max=8000
        )
    
    def load_audio(self, file_path, enhanced_dir, enhanced_name):
        enhanced_file = Path(enhanced_dir) / enhanced_name
        
        if enhanced_file.exists():
            audio, sr = torchaudio.load(enhanced_file)
        else:
            audio, sr = torchaudio.load(file_path)
        
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            audio = resampler(audio)
        
        return audio.squeeze(0)
    
    def augment_audio(self, audio):
        if random.random() > 0.3:
            return audio
            
        # Add light noise
        if random.random() < 0.2:
            noise = torch.randn_like(audio) * 0.005
            audio = audio + noise
        
        # Volume adjustment
        if random.random() < 0.3:
            vol_factor = random.uniform(0.8, 1.2)
            audio = audio * vol_factor
        
        # Time stretching
        if random.random() < 0.2 and len(audio) > 8000:
            stretch_factor = random.uniform(0.95, 1.05)
            new_length = int(len(audio) * stretch_factor)
            indices = torch.linspace(0, len(audio)-1, new_length).long()
            audio = audio[indices]
            
        return audio
    
    def normalize_length(self, audio, augment=False):
        if len(audio) > self.target_length:
            start = random.randint(0, len(audio) - self.target_length) if augment else 0
            audio = audio[start:start + self.target_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.target_length - len(audio)))
        
        return audio
    
    def extract_features(self, audio):
        mfcc = self.mfcc_transform(audio.unsqueeze(0)).squeeze(0).T
        mel = torch.log(self.mel_transform(audio.unsqueeze(0)).squeeze(0) + 1e-8).T
        features = torch.cat([mfcc, mel], dim=1)
        return features