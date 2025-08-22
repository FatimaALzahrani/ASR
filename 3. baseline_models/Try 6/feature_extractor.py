import numpy as np
import librosa
from quality_analyzer import QualityAnalyzer


class FeatureExtractor:
    def __init__(self):
        self.quality_analyzer = QualityAnalyzer()
    
    def _ensure_scalar(self, value):
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            else:
                return float(np.mean(value))
        elif isinstance(value, (list, tuple)):
            return float(np.mean(value))
        else:
            return float(value) if not np.isnan(value) else 0.0
    
    def extract_comprehensive_features(self, y, sr):
        features = {}
        
        features['duration'] = self._ensure_scalar(len(y) / sr)
        features['rms_energy'] = self._ensure_scalar(np.sqrt(np.mean(y**2)))
        features['mean_amplitude'] = self._ensure_scalar(np.mean(np.abs(y)))
        features['std_amplitude'] = self._ensure_scalar(np.std(np.abs(y)))
        features['max_amplitude'] = self._ensure_scalar(np.max(np.abs(y)))
        features['min_amplitude'] = self._ensure_scalar(np.min(np.abs(y)))
        
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
        features['zcr_mean'] = self._ensure_scalar(np.mean(zcr))
        features['zcr_std'] = self._ensure_scalar(np.std(zcr))
        
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = self._ensure_scalar(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = self._ensure_scalar(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = self._ensure_scalar(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = self._ensure_scalar(np.std(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = self._ensure_scalar(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = self._ensure_scalar(np.std(spectral_bandwidth))
        except:
            features['spectral_centroid_mean'] = 0.0
            features['spectral_centroid_std'] = 0.0
            features['spectral_rolloff_mean'] = 0.0
            features['spectral_rolloff_std'] = 0.0
            features['spectral_bandwidth_mean'] = 0.0
            features['spectral_bandwidth_std'] = 0.0
        
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = self._ensure_scalar(np.mean(mfccs[i]))
                features[f'mfcc_{i+1}_std'] = self._ensure_scalar(np.std(mfccs[i]))
        except:
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = 0.0
                features[f'mfcc_{i+1}_std'] = 0.0
        
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            for i in range(12):
                features[f'chroma_{i+1}_mean'] = self._ensure_scalar(np.mean(chroma[i]))
                features[f'chroma_{i+1}_std'] = self._ensure_scalar(np.std(chroma[i]))
        except:
            for i in range(12):
                features[f'chroma_{i+1}_mean'] = 0.0
                features[f'chroma_{i+1}_std'] = 0.0
        
        try:
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            for i in range(13):
                features[f'mel_{i+1}_mean'] = self._ensure_scalar(np.mean(mel_db[i]))
                features[f'mel_{i+1}_std'] = self._ensure_scalar(np.std(mel_db[i]))
        except:
            for i in range(13):
                features[f'mel_{i+1}_mean'] = 0.0
                features[f'mel_{i+1}_std'] = 0.0
        
        features['quality_score'] = self.quality_analyzer.calculate_audio_quality_score(y, sr)
        features['snr'] = self._ensure_scalar(self.quality_analyzer.estimate_snr(y))
        
        return features
