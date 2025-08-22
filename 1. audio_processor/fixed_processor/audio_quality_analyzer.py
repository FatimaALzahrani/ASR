import numpy as np
import librosa
from typing import Dict


class AudioQualityAnalyzerFixed:
    def analyze_audio_quality(self, audio: np.ndarray, sr: int) -> Dict:
        try:
            rms_energy = float(np.sqrt(np.mean(audio**2)))
            
            if len(audio) > sr // 10:
                noise_samples = min(int(0.1 * sr), len(audio) // 4)
                if noise_samples > 0:
                    noise_level = np.sqrt(np.mean(np.concatenate([
                        audio[:noise_samples], audio[-noise_samples:]
                    ])**2))
                    signal_level = np.sqrt(np.mean(audio[noise_samples:-noise_samples]**2))
                    snr = 20 * np.log10((signal_level + 1e-10) / (noise_level + 1e-10))
                else:
                    snr = 15.0
            else:
                snr = 15.0
            
            clipping_ratio = float(np.sum(np.abs(audio) > 0.95) / len(audio))
            
            silence_threshold = max(np.max(np.abs(audio)) * 0.01, 1e-6)
            silence_ratio = float(np.sum(np.abs(audio) < silence_threshold) / len(audio))
            
            try:
                if len(audio) > 1024:
                    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
                    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
                else:
                    spectral_centroid = 1000.0
                    zero_crossing_rate = 0.1
            except:
                spectral_centroid = 1000.0
                zero_crossing_rate = 0.1
            
            quality_score = self.calculate_quality_score(rms_energy, snr, clipping_ratio, silence_ratio)
            
            return {
                'rms_energy': rms_energy,
                'snr_db': float(snr),
                'clipping_ratio': clipping_ratio,
                'silence_ratio': silence_ratio,
                'spectral_centroid': spectral_centroid,
                'zero_crossing_rate': zero_crossing_rate,
                'quality_score': quality_score
            }
            
        except Exception as e:
            print(f"Warning: Error in quality analysis: {e}")
            return {
                'rms_energy': 0.1,
                'snr_db': 15.0,
                'clipping_ratio': 0.0,
                'silence_ratio': 0.3,
                'spectral_centroid': 1000.0,
                'zero_crossing_rate': 0.1,
                'quality_score': 0.6
            }
    
    def calculate_quality_score(self, rms_energy: float, snr: float, 
                              clipping_ratio: float, silence_ratio: float) -> float:
        score = 0.5
        
        if 0.01 < rms_energy < 0.3:
            score += 0.2
        elif rms_energy < 0.005:
            score -= 0.2
        
        if snr > 20:
            score += 0.2
        elif snr > 10:
            score += 0.1
        elif snr < 5:
            score -= 0.2
        
        if clipping_ratio < 0.01:
            score += 0.1
        elif clipping_ratio > 0.05:
            score -= 0.2
        
        if silence_ratio < 0.4:
            score += 0.1
        elif silence_ratio > 0.8:
            score -= 0.2
        
        return float(max(0.0, min(1.0, score)))