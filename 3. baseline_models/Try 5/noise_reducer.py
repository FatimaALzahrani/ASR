import numpy as np
import librosa

class NoiseReducer:
    def __init__(self, noise_reduction_factor=0.5):
        self.noise_reduction_factor = noise_reduction_factor
    
    def remove_noise(self, audio, sr):
        noise_samples = int(0.1 * sr)
        
        if len(audio) > 2 * noise_samples:
            noise_start = audio[:noise_samples]
            noise_end = audio[-noise_samples:]
            noise_profile = np.concatenate([noise_start, noise_end])
        else:
            frame_energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            low_energy_threshold = np.percentile(frame_energy, 20)
            low_energy_frames = frame_energy < low_energy_threshold
            
            if np.any(low_energy_frames):
                hop_length = 512
                noise_indices = np.where(low_energy_frames)[0]
                noise_samples_list = []
                for idx in noise_indices:
                    start = idx * hop_length
                    end = min(start + hop_length, len(audio))
                    noise_samples_list.append(audio[start:end])
                noise_profile = np.concatenate(noise_samples_list) if noise_samples_list else audio[:1000]
            else:
                noise_profile = audio[:1000]
        
        noise_fft = np.fft.fft(noise_profile)
        noise_power = np.abs(noise_fft) ** 2
        audio_fft = np.fft.fft(audio)
        audio_power = np.abs(audio_fft) ** 2
        snr_estimate = audio_power / (noise_power.mean() + 1e-10)
        filter_strength = np.clip(1 - self.noise_reduction_factor / (snr_estimate + 1e-10), 0.1, 1.0)
        filtered_fft = audio_fft * filter_strength
        filtered_audio = np.real(np.fft.ifft(filtered_fft))
        
        return filtered_audio