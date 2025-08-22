import numpy as np
import librosa
from config import Config

class FeatureExtractor:
    def __init__(self):
        self.feature_size = Config.FEATURE_SIZE
        
    def extract_simple_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=Config.AUDIO_SAMPLE_RATE, duration=Config.AUDIO_DURATION)
            
            features = []
            
            rms = np.sqrt(np.mean(y**2))
            features.append(rms)
            
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            features.append(zcr)
            
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            features.append(spectral_centroid)
            
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            features.append(spectral_bandwidth)
            
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            features.append(spectral_rolloff)
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=Config.MFCC_COEFFICIENTS)
            mfcc_means = np.mean(mfccs, axis=1)
            features.extend(mfcc_means)
            
            features.append(np.std(y))
            features.append(np.max(y))
            features.append(np.min(y))
            features.append(len(y) / sr)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return np.zeros(self.feature_size)
    
    def extract_features_batch(self, file_paths):
        features_list = []
        
        print("Extracting audio features...")
        for idx, file_path in enumerate(file_paths):
            features = self.extract_simple_features(file_path)
            features_list.append(features)
            
            if (idx + 1) % Config.PROGRESS_INTERVAL == 0:
                print(f"Processed {idx + 1} files...")
        
        return np.array(features_list)