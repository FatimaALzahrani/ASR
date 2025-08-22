import numpy as np
import scipy.signal
import librosa


class NoiseReducer:
    def simple_noise_reduction(self, audio, sr):
        try:
            nyquist = sr // 2
            cutoff = min(8000, nyquist - 100)
            if cutoff > 100:
                b, a = scipy.signal.butter(5, cutoff / nyquist, btype='low')
                filtered_audio = scipy.signal.filtfilt(b, a, audio)
            else:
                filtered_audio = audio
            
            high_pass_cutoff = 80
            if high_pass_cutoff < nyquist:
                b_hp, a_hp = scipy.signal.butter(3, high_pass_cutoff / nyquist, btype='high')
                filtered_audio = scipy.signal.filtfilt(b_hp, a_hp, filtered_audio)
            
            if len(filtered_audio) > 1024:
                stft = librosa.stft(filtered_audio, n_fft=512, hop_length=160)
                magnitude = np.abs(stft)
                
                noise_threshold = np.percentile(magnitude, 20)
                
                mask = magnitude > noise_threshold
                cleaned_magnitude = magnitude * mask
                
                phase = np.angle(stft)
                cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
                cleaned_audio = librosa.istft(cleaned_stft, hop_length=160)
                
                return cleaned_audio
            else:
                return filtered_audio
            
        except Exception as e:
            print(f"Failed noise reduction: {e}")
            return audio