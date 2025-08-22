import numpy as np

class VolumeNormalizer:
    def __init__(self, target_rms=0.02):
        self.target_rms = target_rms
    
    def normalize(self, audio):
        current_rms = np.sqrt(np.mean(audio**2))
        
        if current_rms > 0:
            normalization_factor = self.target_rms / current_rms
            normalized_audio = audio * normalization_factor
            max_val = np.max(np.abs(normalized_audio))
            if max_val > 0.95:
                normalized_audio = normalized_audio * (0.95 / max_val)
        else:
            normalized_audio = audio
            
        return normalized_audio