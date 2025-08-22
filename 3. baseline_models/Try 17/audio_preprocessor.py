import numpy as np
import librosa
from scipy import signal

class AudioPreprocessor:
    def __init__(self):
        pass
    
    def enhanced_audio_preprocessing(self, y, sr, speaker_profile):
        try:
            clarity = speaker_profile.get("clarity", 0.5)
            
            y = librosa.util.normalize(y)
            y = self.advanced_silence_removal(y, sr)
            
            if clarity < 0.6:
                y = self.enhance_unclear_speech(y, sr)
            elif clarity > 0.8:
                y = self.enhance_clear_speech(y, sr)
            
            y = librosa.util.normalize(y)
            return y
        except:
            return librosa.util.normalize(y)
    
    def advanced_silence_removal(self, y, sr):
        try:
            frame_length = 2048
            hop_length = 512
            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            energy_threshold = np.percentile(energy, 30)
            active_frames = energy > energy_threshold
            
            if np.any(active_frames):
                active_samples = librosa.frames_to_samples(np.where(active_frames)[0], hop_length=hop_length)
                
                if len(active_samples) > 0:
                    start_sample = max(0, active_samples[0] - hop_length)
                    end_sample = min(len(y), active_samples[-1] + hop_length)
                    y = y[start_sample:end_sample]
            
            return y
        except:
            return y
    
    def enhance_unclear_speech(self, y, sr):
        try:
            y = np.sign(y) * np.power(np.abs(y), 0.7)
            
            sos = signal.butter(6, 120, btype='high', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            sos = signal.butter(6, 7000, btype='low', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            return y
        except:
            return y
    
    def enhance_clear_speech(self, y, sr):
        try:
            sos = signal.butter(4, 80, btype='high', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            sos = signal.butter(4, 8000, btype='low', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            return y
        except:
            return y