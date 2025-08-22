import numpy as np
import librosa
from typing import Dict


class AudioQualityAnalyzer:    
    def __init__(self):
        self.default_metrics = {
            'rms_energy': 0.1, 
            'snr_db': 10.0, 
            'clipping_ratio': 0.0,
            'silence_ratio': 0.5, 
            'spectral_centroid': 1000.0,
            'spectral_bandwidth': 500.0, 
            'spectral_rolloff': 2000.0,
            'zero_crossing_rate': 0.1, 
            'quality_score': 0.5
        }
    
    def analyze_audio_quality(self, audio: np.ndarray, sr: int) -> Dict:
        try:
            # 1. RMS Energy
            rms_energy = np.sqrt(np.mean(audio**2))
            
            # 2. Signal-to-Noise Ratio estimation
            snr = self._estimate_snr(audio, sr)
            
            # 3. Clipping detection
            clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
            
            # 4. Silence ratio
            silence_threshold = np.max(np.abs(audio)) * 0.01
            silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
            
            # 5. Spectral features
            spectral_features = self._extract_spectral_features(audio, sr)
            
            # 6. Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # 7. Overall quality score
            quality_score = self._calculate_quality_score(
                rms_energy, snr, clipping_ratio, silence_ratio
            )
            
            return {
                'rms_energy': float(rms_energy),
                'snr_db': float(snr),
                'clipping_ratio': float(clipping_ratio),
                'silence_ratio': float(silence_ratio),
                'zero_crossing_rate': float(zcr),
                'quality_score': float(quality_score),
                **spectral_features
            }
            
        except Exception as e:
            print(f"Warning: Error in quality analysis: {e}")
            return self.default_metrics.copy()
    
    def _estimate_snr(self, audio: np.ndarray, sr: int) -> float:
        noise_samples = min(int(0.1 * sr), len(audio) // 4)
        
        if len(audio) > 2 * noise_samples:
            noise_level = np.sqrt(np.mean(np.concatenate([
                audio[:noise_samples], audio[-noise_samples:]
            ])**2))
            signal_level = np.sqrt(np.mean(audio[noise_samples:-noise_samples]**2))
            snr = 20 * np.log10((signal_level + 1e-10) / (noise_level + 1e-10))
        else:
            snr = 10.0
        
        return snr
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        if len(audio) > 1024:
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        else:
            spectral_centroid = 1000.0
            spectral_bandwidth = 500.0
            spectral_rolloff = 2000.0
        
        return {
            'spectral_centroid': float(spectral_centroid),
            'spectral_bandwidth': float(spectral_bandwidth),
            'spectral_rolloff': float(spectral_rolloff)
        }
    
    def _calculate_quality_score(self, rms_energy: float, snr: float, 
                               clipping_ratio: float, silence_ratio: float) -> float:
        score = 0.5  # Base score
        
        # Energy assessment
        if 0.01 < rms_energy < 0.3:
            score += 0.2
        elif rms_energy < 0.005:
            score -= 0.3
        
        # SNR assessment
        if snr > 20:
            score += 0.2
        elif snr > 10:
            score += 0.1
        elif snr < 5:
            score -= 0.3
        
        # Clipping assessment
        if clipping_ratio < 0.01:
            score += 0.1
        elif clipping_ratio > 0.05:
            score -= 0.2
        
        # Silence assessment
        if silence_ratio < 0.3:
            score += 0.1
        elif silence_ratio > 0.7:
            score -= 0.2
        
        return max(0.0, min(1.0, score))