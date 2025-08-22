import numpy as np


class GentleVolumeNormalizer:
    def gentle_normalization(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        try:
            audio = audio - np.mean(audio)
            
            current_rms = np.sqrt(np.mean(audio**2))
            if current_rms > 1e-6:
                normalization_factor = target_rms / current_rms
                
                if normalization_factor > 3.0:
                    normalization_factor = 3.0
                elif normalization_factor < 0.3:
                    normalization_factor = 0.3
                
                audio = audio * normalization_factor
            
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0.9:
                compression_ratio = 0.9 / max_amplitude
                audio = audio * compression_ratio
            
            audio = np.tanh(audio * 0.9) * 0.95
            
            return audio
            
        except Exception as e:
            print(f"Warning: Gentle normalization failed: {e}")
            return audio