import numpy as np
import librosa


class QualityAnalyzer:
    def __init__(self):
        pass
    
    def calculate_audio_quality_score(self, y, sr):
        try:
            quality_factors = {}
            
            snr = self.estimate_snr(y)
            quality_factors['snr'] = max(0, min((snr + 10) / 50, 1.0))
            
            dynamic_range = np.max(np.abs(y)) - np.min(np.abs(y))
            quality_factors['dynamic_range'] = min(dynamic_range, 1.0)
            
            clipping_rate = np.sum(np.abs(y) > 0.99) / len(y)
            quality_factors['clipping'] = 1.0 - clipping_rate
            
            silence_rate = np.sum(np.abs(y) < 0.01) / len(y)
            quality_factors['silence'] = 1.0 - min(silence_rate, 0.8) / 0.8
            
            duration = len(y) / sr
            if 0.5 <= duration <= 5.0:
                quality_factors['duration'] = 1.0
            elif duration < 0.5:
                quality_factors['duration'] = duration / 0.5
            else:
                quality_factors['duration'] = max(0.1, 1.0 - (duration - 5.0) / 10.0)
            
            weights = {
                'snr': 0.4,
                'dynamic_range': 0.2,
                'clipping': 0.2,
                'silence': 0.1,
                'duration': 0.1
            }
            
            quality_score = sum(quality_factors[factor] * weights[factor] 
                              for factor in quality_factors)
            
            return max(0.0, min(quality_score, 1.0))
            
        except Exception as e:
            print(f"Quality calculation failed: {str(e)}")
            return 0.5
    
    def estimate_snr(self, signal):
        try:
            signal_power = np.mean(signal**2)
            
            frame_energy = librosa.feature.rms(y=signal, frame_length=2048, hop_length=512)[0]
            noise_threshold = np.percentile(frame_energy, 10)
            noise_frames = frame_energy < noise_threshold
            
            if np.sum(noise_frames) > 0:
                noise_segments = []
                for i, is_noise in enumerate(noise_frames):
                    if is_noise:
                        start = i * 512
                        end = min(start + 512, len(signal))
                        noise_segments.extend(signal[start:end])
                
                if len(noise_segments) > 0:
                    noise_power = np.mean(np.array(noise_segments)**2)
                else:
                    noise_power = signal_power * 0.01
            else:
                noise_power = signal_power * 0.01
            
            if noise_power == 0:
                return 40.0
            
            snr = 10 * np.log10(signal_power / noise_power)
            return max(snr, -10.0)
            
        except:
            return 20.0
