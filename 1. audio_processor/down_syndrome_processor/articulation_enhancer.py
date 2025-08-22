import numpy as np
import scipy.signal


class ArticulationEnhancer:
    def __init__(self, down_syndrome_params):
        self.down_syndrome_params = down_syndrome_params
    
    def enhance_articulation(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            if not self.down_syndrome_params['articulation_enhancement']:
                return audio
            
            if self.down_syndrome_params['low_freq_emphasis']:
                nyquist = sr // 2
                low_freq = 1000 / nyquist
                high_freq = 4000 / nyquist
                
                if high_freq < 0.95 and low_freq > 0.01:
                    b, a = scipy.signal.butter(2, [low_freq, high_freq], btype='band')
                    enhanced_band = scipy.signal.filtfilt(b, a, audio)
                    
                    audio = audio + 0.3 * enhanced_band
            
            threshold = np.percentile(np.abs(audio), 20)
            weak_indices = np.abs(audio) < threshold
            audio[weak_indices] *= 1.5
            
            if len(audio) > 1:
                pre_emphasis_coeff = 0.95
                audio = np.append(audio[0], audio[1:] - pre_emphasis_coeff * audio[:-1])
            
            return audio
            
        except Exception as e:
            print(f"Warning: Articulation enhancement failed: {e}")
            return audio