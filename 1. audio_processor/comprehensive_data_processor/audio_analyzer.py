import librosa
import numpy as np


class AudioAnalyzer:
    def analyze_file(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=None)
            
            return {
                'duration': len(y) / sr,
                'sample_rate': sr,
                'channels': 1 if len(y.shape) == 1 else y.shape[0],
                'rms_energy': np.sqrt(np.mean(y**2)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            }
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def estimate_quality(self, audio_info):
        score = 0.5
        
        if audio_info['rms_energy'] > 0.01:
            score += 0.2
        elif audio_info['rms_energy'] < 0.001:
            score -= 0.2
            
        if 0.05 < audio_info['zero_crossing_rate'] < 0.15:
            score += 0.1
        else:
            score -= 0.1
            
        if 1000 < audio_info['spectral_centroid'] < 3000:
            score += 0.1
        else:
            score -= 0.1
            
        if 1.0 < audio_info['duration'] < 5.0:
            score += 0.1
        else:
            score -= 0.1
            
        return max(0.0, min(1.0, score))