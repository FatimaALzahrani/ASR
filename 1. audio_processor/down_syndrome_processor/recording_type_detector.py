import numpy as np
import librosa


class RecordingTypeDetector:
    def detect_recording_type(self, audio: np.ndarray, sr: int) -> str:
        try:
            stft = np.abs(librosa.stft(audio, n_fft=512))
            
            high_freq_energy = np.mean(stft[stft.shape[0]//2:, :])
            low_freq_energy = np.mean(stft[:stft.shape[0]//2, :])
            
            freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)
            
            if freq_ratio > 0.3:
                return 'microphone'
            else:
                return 'computer'
                
        except:
            return 'unknown'