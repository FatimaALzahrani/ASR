#!/usr/bin/env python3

import numpy as np
import librosa
from scipy import signal

class AudioProcessor:
    
    def enhanced_audio_preprocessing_v2(self, y, sr, speaker_profile, word_info=None):
        try:
            clarity = speaker_profile.get("clarity", 0.5)
            age_group = speaker_profile.get("age", "0").split("-")[0]
            age = int(age_group) if age_group.isdigit() else 5
            
            y = librosa.util.normalize(y)
            
            if age < 10:
                y = self.process_young_child_speech(y, sr)
            elif age > 20:
                y = self.process_adult_speech(y, sr)
            
            y = self.advanced_silence_removal_v2(y, sr)
            
            if clarity < 0.5:
                y = self.enhance_very_unclear_speech(y, sr)
            elif clarity < 0.7:
                y = self.enhance_unclear_speech(y, sr)
            elif clarity > 0.85:
                y = self.enhance_very_clear_speech(y, sr)
            else:
                y = self.enhance_clear_speech(y, sr)
            
            if word_info:
                difficulty = word_info.get('difficulty', 'medium')
                if difficulty in ['hard', 'very_hard']:
                    y = self.enhance_for_difficult_words(y, sr)
            
            y = librosa.util.normalize(y)
            
            return y
        except Exception as e:
            print(f"Audio processing error: {e}")
            return librosa.util.normalize(y)
    
    def process_young_child_speech(self, y, sr):
        try:
            sos = signal.butter(4, 150, btype='high', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            sos = signal.butter(6, 6000, btype='low', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            return y
        except:
            return y
    
    def process_adult_speech(self, y, sr):
        try:
            sos = signal.butter(4, 100, btype='high', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            sos = signal.butter(4, 7500, btype='low', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            return y
        except:
            return y
    
    def advanced_silence_removal_v2(self, y, sr):
        try:
            frame_lengths = [1024, 2048, 4096]
            hop_length = 256
            
            energy_measures = []
            for frame_length in frame_lengths:
                try:
                    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                    energy_measures.append(energy)
                except:
                    continue
            
            if not energy_measures:
                return y
            
            combined_energy = np.mean(energy_measures, axis=0)
            
            energy_threshold = np.percentile(combined_energy, 25)
            
            active_frames = combined_energy > energy_threshold
            
            if np.any(active_frames):
                active_samples = librosa.frames_to_samples(np.where(active_frames)[0], hop_length=hop_length)
                
                if len(active_samples) > 0:
                    start_sample = max(0, active_samples[0] - hop_length * 2)
                    end_sample = min(len(y), active_samples[-1] + hop_length * 2)
                    y = y[start_sample:end_sample]
            
            return y
        except:
            return y
    
    def enhance_unclear_speech(self, y, sr):
        try:
            y = np.sign(y) * np.power(np.abs(y), 0.7)
            
            sos = signal.butter(6, 120, btype='high', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            sos = signal.butter(6, 7000, btype='low', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            return y
        except:
            return y
    
    def enhance_clear_speech(self, y, sr):
        try:
            sos = signal.butter(4, 80, btype='high', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            sos = signal.butter(4, 8000, btype='low', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            return y
        except:
            return y

    def enhance_very_unclear_speech(self, y, sr):
        try:
            y = np.sign(y) * np.power(np.abs(y), 0.6)
            
            sos = signal.butter(8, 150, btype='high', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            sos = signal.butter(8, 6000, btype='low', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            y = np.tanh(y * 2) / 2
            
            return y
        except:
            return y
    
    def enhance_very_clear_speech(self, y, sr):
        try:
            sos = signal.butter(2, 60, btype='high', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            sos = signal.butter(2, 9000, btype='low', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            return y
        except:
            return y
    
    def enhance_for_difficult_words(self, y, sr):
        try:
            y = np.sign(y) * np.power(np.abs(y), 0.8)
            
            sos = signal.butter(6, [100, 8000], btype='band', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
            
            return y
        except:
            return y
