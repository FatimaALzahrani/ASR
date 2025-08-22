import numpy as np
import librosa
from scipy.ndimage import median_filter

class SpeechBoundaryDetector:
    def __init__(self, frame_length=2048, hop_length=512):
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def detect_boundaries(self, audio, sr):
        frames = librosa.util.frame(audio, frame_length=self.frame_length, hop_length=self.hop_length)
        frame_energy = np.sum(frames**2, axis=0)
        smoothed_energy = median_filter(frame_energy, size=5)
        silence_threshold = np.percentile(smoothed_energy, 25)
        speech_frames = smoothed_energy > silence_threshold
        
        if np.any(speech_frames):
            speech_indices = np.where(speech_frames)[0]
            start_frame = speech_indices[0]
            end_frame = speech_indices[-1]
            start_time = start_frame * self.hop_length / sr
            end_time = (end_frame + 1) * self.hop_length / sr
            margin = 0.05
            start_time = max(0, start_time - margin)
            end_time = min(len(audio) / sr, end_time + margin)
        else:
            start_time = 0
            end_time = len(audio) / sr
            
        return start_time, end_time