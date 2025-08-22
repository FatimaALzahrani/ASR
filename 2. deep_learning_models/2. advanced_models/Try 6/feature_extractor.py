import numpy as np
import librosa
from pathlib import Path
from typing import Optional

class EnhancedFeatureExtractor:
    def __init__(self, sr: int = 16000, n_mfcc: int = 13, n_mels: int = 20):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.expected_feature_length = None
    
    def extract_comprehensive_features(self, audio_file: Path) -> Optional[np.ndarray]:
        try:
            y, sr = librosa.load(str(audio_file), sr=self.sr)
            
            if len(y) == 0:
                return None
            
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            if len(y_trimmed) == 0:
                y_trimmed = y
                
            y = librosa.util.normalize(y_trimmed)
            y = librosa.effects.preemphasis(y)
            
            feature_list = []
            
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=2048, hop_length=512)
                if mfcc.shape[1] > 0:
                    feature_list.extend(np.mean(mfcc, axis=1))
                    feature_list.extend(np.std(mfcc, axis=1))
                    feature_list.extend(np.max(mfcc, axis=1))
                    feature_list.extend(np.min(mfcc, axis=1))
                    feature_list.extend(np.median(mfcc, axis=1))
                    
                    mfcc_delta = librosa.feature.delta(mfcc)
                    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                    feature_list.extend(np.mean(mfcc_delta, axis=1))
                    feature_list.extend(np.mean(mfcc_delta2, axis=1))
                else:
                    feature_list.extend([0.0] * (self.n_mfcc * 7))
            except:
                feature_list.extend([0.0] * (self.n_mfcc * 7))
            
            try:
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, n_fft=2048, hop_length=512)
                if mel_spec.shape[1] > 0:
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    feature_list.extend(np.mean(mel_spec_db, axis=1))
                    feature_list.extend(np.std(mel_spec_db, axis=1))
                    feature_list.extend(np.max(mel_spec_db, axis=1))
                    feature_list.extend(np.min(mel_spec_db, axis=1))
                else:
                    feature_list.extend([0.0] * (self.n_mels * 4))
            except:
                feature_list.extend([0.0] * (self.n_mels * 4))
            
            spectral_features = []
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_features.extend([np.mean(spectral_centroids), np.std(spectral_centroids), np.max(spectral_centroids)])
            except:
                spectral_features.extend([0.0, 0.0, 0.0])
            
            try:
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                spectral_features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff), np.max(spectral_rolloff)])
            except:
                spectral_features.extend([0.0, 0.0, 0.0])
            
            try:
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                spectral_features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth), np.max(spectral_bandwidth)])
            except:
                spectral_features.extend([0.0, 0.0, 0.0])
            
            try:
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                spectral_features.extend([np.mean(spectral_contrast), np.std(spectral_contrast)])
            except:
                spectral_features.extend([0.0, 0.0])
            
            try:
                spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
                spectral_features.extend([np.mean(spectral_flatness), np.std(spectral_flatness)])
            except:
                spectral_features.extend([0.0, 0.0])
            
            try:
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
                spectral_features.extend([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)])
            except:
                spectral_features.extend([0.0, 0.0])
            
            feature_list.extend(spectral_features)
            
            try:
                rms_energy = librosa.feature.rms(y=y)[0]
                feature_list.extend([np.mean(rms_energy), np.std(rms_energy), np.max(rms_energy)])
            except:
                feature_list.extend([0.0, 0.0, 0.0])
            
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                feature_list.extend([np.mean(chroma), np.std(chroma)])
            except:
                feature_list.extend([0.0, 0.0])
            
            try:
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                feature_list.extend([np.mean(tonnetz), np.std(tonnetz)])
            except:
                feature_list.extend([0.0, 0.0])
            
            try:
                f0 = librosa.yin(y, fmin=50, fmax=400)
                f0_clean = f0[f0 > 0]
                if len(f0_clean) > 0:
                    feature_list.extend([
                        np.mean(f0_clean),
                        np.std(f0_clean),
                        np.max(f0_clean),
                        np.min(f0_clean),
                        len(f0_clean) / len(f0)
                    ])
                else:
                    feature_list.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            except:
                feature_list.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
            try:
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                
                feature_list.extend([
                    len(onset_frames) / len(y) * sr if len(y) > 0 else 0.0,
                    float(tempo) if tempo else 0.0,
                    len(beats) / len(y) * sr if len(beats) > 0 and len(y) > 0 else 0.0
                ])
            except:
                feature_list.extend([0.0, 0.0, 0.0])
            
            feature_vector = np.array(feature_list, dtype=np.float32)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            if self.expected_feature_length is None:
                self.expected_feature_length = len(feature_vector)
            
            if len(feature_vector) != self.expected_feature_length:
                if len(feature_vector) < self.expected_feature_length:
                    padding = self.expected_feature_length - len(feature_vector)
                    feature_vector = np.pad(feature_vector, (0, padding), 'constant')
                else:
                    feature_vector = feature_vector[:self.expected_feature_length]
            
            return feature_vector
            
        except Exception as e:
            if self.expected_feature_length is not None:
                return np.zeros(self.expected_feature_length, dtype=np.float32)
            return None