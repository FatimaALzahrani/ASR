import numpy as np


class VolumeNormalizer:
    def normalize_volume(self, audio):
        try:
            audio = audio - np.mean(audio)
            
            rms = np.sqrt(np.mean(audio**2))
            if rms > 1e-6:
                target_rms = 0.1
                audio = audio * (target_rms / rms)
            
            max_val = np.max(np.abs(audio))
            if max_val > 0.95:
                audio = audio * (0.95 / max_val)
            
            threshold = 0.5
            ratio = 3.0
            above_threshold = np.abs(audio) > threshold
            if np.any(above_threshold):
                audio[above_threshold] = np.sign(audio[above_threshold]) * (
                    threshold + (np.abs(audio[above_threshold]) - threshold) / ratio
                )
            
            return audio
            
        except Exception as e:
            print(f"Failed volume normalization: {e}")
            return audio