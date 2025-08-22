import numpy as np
import scipy.signal
import librosa
from scipy.ndimage import uniform_filter1d


class AdvancedNoiseProcessor:
    def __init__(self, down_syndrome_params):
        self.down_syndrome_params = down_syndrome_params
    
    def advanced_noise_reduction(self, audio: np.ndarray, sr: int, 
                               recording_type: str, strength: float = 0.5) -> np.ndarray:
        try:
            if recording_type == 'microphone':
                audio = self.reduce_microphone_noise(audio, sr, strength)
            else:
                audio = self.reduce_computer_artifacts(audio, sr, strength * 0.7)
            
            return audio
            
        except Exception as e:
            print(f"Warning: Noise reduction failed: {e}")
            return audio
    
    def reduce_microphone_noise(self, audio: np.ndarray, sr: int, strength: float) -> np.ndarray:
        nyquist = sr // 2
        low_cutoff = 80 / nyquist
        high_cutoff = min(8000, nyquist - 100) / nyquist
        
        if high_cutoff > low_cutoff:
            b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            audio = scipy.signal.filtfilt(b, a, audio)
        
        if len(audio) > 1024:
            audio = self.spectral_subtraction(audio, sr, strength)
        
        if self.down_syndrome_params['breathing_noise_reduction']:
            b_hp, a_hp = scipy.signal.butter(2, 120 / nyquist, btype='high')
            audio = scipy.signal.filtfilt(b_hp, a_hp, audio)
        
        return audio
    
    def reduce_computer_artifacts(self, audio: np.ndarray, sr: int, strength: float) -> np.ndarray:
        audio = audio - np.mean(audio)
        
        nyquist = sr // 2
        cutoff = min(7000, nyquist - 100) / nyquist
        
        if cutoff > 0.1:
            b, a = scipy.signal.butter(3, cutoff, btype='low')
            audio = scipy.signal.filtfilt(b, a, audio)
        
        if strength > 0.5:
            audio = uniform_filter1d(audio, size=3)
        
        return audio
    
    def spectral_subtraction(self, audio: np.ndarray, sr: int, strength: float) -> np.ndarray:
        try:
            stft = librosa.stft(audio, n_fft=512, hop_length=160)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            noise_frames = max(1, int(0.1 * stft.shape[1]))
            noise_spectrum = np.mean(np.concatenate([
                magnitude[:, :noise_frames],
                magnitude[:, -noise_frames:]
            ], axis=1), axis=1, keepdims=True)
            
            alpha = strength * 2.0
            cleaned_magnitude = magnitude - alpha * noise_spectrum
            
            cleaned_magnitude = np.maximum(cleaned_magnitude, 0.1 * magnitude)
            
            cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
            cleaned_audio = librosa.istft(cleaned_stft, hop_length=160)
            
            return cleaned_audio
            
        except:
            return audio