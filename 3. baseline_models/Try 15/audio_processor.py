import numpy as np
import librosa
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class AdvancedAudioProcessor:
    def __init__(self, target_sr=16000, target_duration=30.0):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(target_sr * target_duration)
        
    def load_and_preprocess(self, audio_path: str) -> np.ndarray:
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            audio = librosa.util.normalize(audio)
            audio = self.remove_silence(audio)
            audio = self.enhance_audio(audio)
            audio = self.adjust_length(audio)
            return audio
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros(self.target_length)
    
    def remove_silence(self, audio: np.ndarray, threshold=0.01) -> np.ndarray:
        if len(audio) == 0:
            return audio
            
        energy = np.abs(audio)
        start_idx = 0
        end_idx = len(audio)
        
        for i, val in enumerate(energy):
            if val > threshold:
                start_idx = max(0, i - int(0.1 * self.target_sr))
                break
                
        for i in range(len(energy)-1, -1, -1):
            if energy[i] > threshold:
                end_idx = min(len(audio), i + int(0.1 * self.target_sr))
                break
        
        return audio[start_idx:end_idx] if end_idx > start_idx else audio
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio
            
        try:
            sos = signal.butter(5, 80, btype='high', fs=self.target_sr, output='sos')
            audio = signal.sosfilt(sos, audio)
            
            sos = signal.butter(5, 8000, btype='low', fs=self.target_sr, output='sos')
            audio = signal.sosfilt(sos, audio)
            
            audio = librosa.util.normalize(audio)
            
            return audio
        except:
            return audio
    
    def adjust_length(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) > self.target_length:
            start = (len(audio) - self.target_length) // 2
            audio = audio[start:start + self.target_length]
        elif len(audio) < self.target_length:
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio