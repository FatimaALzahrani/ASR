import numpy as np
import librosa


class AudioQualityAnalyzer:
    def analyze_audio_quality(self, audio, sr):
        try:
            rms_energy = np.sqrt(np.mean(audio**2))
            
            noise_samples = int(0.1 * sr)
            if len(audio) > 2 * noise_samples:
                noise_start = audio[:noise_samples]
                noise_end = audio[-noise_samples:]
                noise_level = np.sqrt(np.mean(np.concatenate([noise_start, noise_end])**2))
                
                signal_start = int(0.2 * sr)
                signal_end = int(0.8 * sr) if len(audio) > int(0.8 * sr) else len(audio)
                if signal_end > signal_start:
                    signal_level = np.sqrt(np.mean(audio[signal_start:signal_end]**2))
                else:
                    signal_level = rms_energy
                
                snr = 20 * np.log10(signal_level / (noise_level + 1e-10))
            else:
                snr = 10.0
            
            clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
            
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
            
            if len(audio) > 512:
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
            else:
                spectral_centroid = 1000.0
                spectral_bandwidth = 500.0
            
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            return {
                'rms_energy': float(rms_energy),
                'snr': float(snr),
                'clipping_ratio': float(clipping_ratio),
                'silence_ratio': float(silence_ratio),
                'spectral_centroid': float(spectral_centroid),
                'spectral_bandwidth': float(spectral_bandwidth),
                'zero_crossing_rate': float(zcr),
                'duration': len(audio) / sr
            }
            
        except Exception as e:
            print(f"Error in quality analysis: {e}")
            return {
                'rms_energy': 0.1,
                'snr': 10.0,
                'clipping_ratio': 0.0,
                'silence_ratio': 0.5,
                'spectral_centroid': 1000.0,
                'spectral_bandwidth': 500.0,
                'zero_crossing_rate': 0.1,
                'duration': len(audio) / sr
            }