import torch
import numpy as np
import librosa
from typing import Optional
from transformers import WhisperProcessor
from config import Config

class AudioProcessor:
    def __init__(self, processor: WhisperProcessor, device: torch.device):
        self.processor = processor
        self.device = device
        
    def preprocess_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        try:
            audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
            
            if len(audio) == 0:
                return None
            
            audio = audio / np.max(np.abs(audio))
            
            inputs = self.processor(
                audio, 
                sampling_rate=Config.SAMPLE_RATE, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            return inputs.input_features.to(self.device)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None