import numpy as np
import librosa


class SmartDurationManager:
    def __init__(self, target_length):
        self.target_length = target_length
    
    def smart_duration_adjustment(self, audio: np.ndarray, sr: int, 
                                silence_threshold: float = 0.02) -> np.ndarray:
        try:
            current_length = len(audio)
            
            if current_length > self.target_length:
                audio = self.smart_trim(audio, sr, silence_threshold)
            elif current_length < self.target_length:
                audio = self.smart_padding(audio, sr)
            
            return audio
            
        except Exception as e:
            print(f"Warning: Duration adjustment failed: {e}")
            return audio[:self.target_length] if len(audio) > self.target_length else np.pad(audio, (0, self.target_length - len(audio)))
    
    def smart_trim(self, audio: np.ndarray, sr: int, silence_threshold: float) -> np.ndarray:
        frame_length = int(0.025 * sr)
        hop_length = int(0.01 * sr)
        
        frames = librosa.util.frame(audio, frame_length=frame_length, 
                                  hop_length=hop_length, axis=0)
        frame_energy = np.mean(frames**2, axis=0)
        
        active_frames = frame_energy > silence_threshold
        
        if np.any(active_frames):
            active_indices = np.where(active_frames)[0]
            start_frame = max(0, active_indices[0] - 2)
            end_frame = min(len(frame_energy), active_indices[-1] + 3)
            
            start_sample = start_frame * hop_length
            end_sample = min(len(audio), end_frame * hop_length + frame_length)
            
            active_audio = audio[start_sample:end_sample]
            
            if len(active_audio) > self.target_length:
                excess = len(active_audio) - self.target_length
                start_cut = excess // 2
                active_audio = active_audio[start_cut:start_cut + self.target_length]
            
            return active_audio
        else:
            start_idx = (len(audio) - self.target_length) // 2
            return audio[start_idx:start_idx + self.target_length]
    
    def smart_padding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        padding_needed = self.target_length - len(audio)
        
        if len(audio) < self.target_length // 4:
            repeats_needed = (self.target_length // len(audio)) + 1
            repeated_audio = np.tile(audio, repeats_needed)
            return repeated_audio[:self.target_length]
        else:
            pad_start = padding_needed // 2
            pad_end = padding_needed - pad_start
            
            fade_length = min(int(0.01 * sr), len(audio) // 10)
            
            if fade_length > 0:
                fade_in = np.linspace(0, 1, fade_length)
                fade_out = np.linspace(1, 0, fade_length)
                
                audio[:fade_length] *= fade_in
                audio[-fade_length:] *= fade_out
            
            padded_audio = np.pad(audio, (pad_start, pad_end), mode='constant', constant_values=0)
            
            return padded_audio