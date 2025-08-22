import numpy as np
import librosa
import librosa.effects


class AudioProcessor:
    def __init__(self, sample_rate=22050, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
    
    def enhance_audio_signal(self, y, sr):
        y_denoised = self.spectral_gating_denoise(y, sr)
        y_normalized = librosa.util.normalize(y_denoised)
        y_preemphasized = self.apply_preemphasis(y_normalized)
        y_trimmed, _ = librosa.effects.trim(y_preemphasized, top_db=20)
        
        target_length = int(self.duration * sr)
        if len(y_trimmed) > target_length:
            start = (len(y_trimmed) - target_length) // 2
            y_final = y_trimmed[start:start + target_length]
        else:
            pad_length = target_length - len(y_trimmed)
            y_final = np.pad(y_trimmed, (0, pad_length), mode='constant')
        
        return y_final
    
    def spectral_gating_denoise(self, y, sr, alpha=2.0, beta=0.15):
        try:
            stft = librosa.stft(y, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            noise_frames = min(10, magnitude.shape[1])
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            
            spectral_floor = beta * magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, spectral_floor)
            
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            y_enhanced = librosa.istft(enhanced_stft, hop_length=512)
            
            return y_enhanced
        except:
            return y
    
    def apply_preemphasis(self, y, coeff=0.97):
        try:
            return np.append(y[0], y[1:] - coeff * y[:-1])
        except:
            return y
