import numpy as np


class EnhancementMetricsCalculator:
    def calculate_enhancement_metrics(self, original: np.ndarray, enhanced: np.ndarray, sr: int) -> dict:
        try:
            original_rms = np.sqrt(np.mean(original**2))
            enhanced_rms = np.sqrt(np.mean(enhanced**2))
            
            def estimate_snr(audio):
                if len(audio) < sr // 10:
                    return 10.0
                    
                noise_samples = min(int(0.1 * sr), len(audio) // 4)
                noise_level = np.sqrt(np.mean(np.concatenate([
                    audio[:noise_samples], audio[-noise_samples:]
                ])**2))
                signal_level = np.sqrt(np.mean(audio[noise_samples:-noise_samples]**2))
                
                return 20 * np.log10((signal_level + 1e-10) / (noise_level + 1e-10))
            
            original_snr = estimate_snr(original)
            enhanced_snr = estimate_snr(enhanced)
            
            original_clipping = np.sum(np.abs(original) > 0.95) / len(original)
            enhanced_clipping = np.sum(np.abs(enhanced) > 0.95) / len(enhanced)
            
            return {
                'rms_improvement': enhanced_rms / (original_rms + 1e-10),
                'snr_improvement': enhanced_snr - original_snr,
                'clipping_reduction': original_clipping - enhanced_clipping,
                'original_snr': original_snr,
                'enhanced_snr': enhanced_snr
            }
            
        except:
            return {
                'rms_improvement': 1.0,
                'snr_improvement': 0.0,
                'clipping_reduction': 0.0,
                'original_snr': 10.0,
                'enhanced_snr': 10.0
            }