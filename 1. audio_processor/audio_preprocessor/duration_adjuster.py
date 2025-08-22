import numpy as np


class DurationAdjuster:
    def __init__(self, target_length):
        self.target_length = target_length
    
    def adjust_duration(self, audio, sr):
        try:
            current_length = len(audio)
            
            if current_length > self.target_length:
                start_idx = (current_length - self.target_length) // 2
                audio = audio[start_idx:start_idx + self.target_length]
                
            elif current_length < self.target_length:
                padding_needed = self.target_length - current_length
                
                if current_length < self.target_length // 4:
                    repeats = (self.target_length // current_length) + 1
                    audio = np.tile(audio, repeats)[:self.target_length]
                else:
                    pad_start = padding_needed // 2
                    pad_end = padding_needed - pad_start
                    audio = np.pad(audio, (pad_start, pad_end), mode='constant', constant_values=0)
            
            return audio
            
        except Exception as e:
            print(f"Failed duration adjustment: {e}")
            return audio