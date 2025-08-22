import numpy as np


class SpeechEnhancer:
    def enhance_speech(self, audio, sr):
        try:
            pre_emphasis = 0.97
            if len(audio) > 1:
                emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            else:
                emphasized = audio
            
            threshold_low = 0.05
            gain = 2.0
            
            low_amplitude = np.abs(emphasized) < threshold_low
            emphasized[low_amplitude] = emphasized[low_amplitude] * gain
            
            max_val = np.max(np.abs(emphasized))
            if max_val > 0.95:
                emphasized = emphasized * (0.95 / max_val)
            
            return emphasized
            
        except Exception as e:
            print(f"Failed speech enhancement: {e}")
            return audio