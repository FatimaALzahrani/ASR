import numpy as np
import librosa
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    def __init__(self):
        self.sample_rate = 22050
        self.max_duration = 3.0
        self.n_mfcc = 20
        self.n_mels = 30
        self.n_fft = 1024
        self.hop_length = 256
        self.n_chroma = 12
        self.lpc_order = 12
    
    def extract_features(self, audio_file):
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate, duration=self.max_duration)
            
            if len(y) == 0:
                return None
                
            y = librosa.util.normalize(y)
            features = {}
            
            features.update(self._extract_mfcc_features(y, sr))
            features.update(self._extract_spectral_features(y, sr))
            features.update(self._extract_prosodic_features(y, sr))
            features.update(self._extract_energy_features(y, sr))
            features.update(self._extract_chroma_features(y, sr))
            features.update(self._extract_temporal_features(y, sr))
            features.update(self._extract_lpc_features(y))
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_file}: {e}")
            return None
    
    def _extract_mfcc_features(self, y, sr):
        features = {}
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, 
                                      n_fft=self.n_fft, hop_length=self.hop_length)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            for i in range(self.n_mfcc):
                features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
                features[f'mfcc_{i}_std'] = np.std(mfcc[i])
                features[f'mfcc_{i}_max'] = np.max(mfcc[i])
                features[f'mfcc_{i}_min'] = np.min(mfcc[i])
                features[f'mfcc_{i}_median'] = np.median(mfcc[i])
                features[f'mfcc_{i}_range'] = np.max(mfcc[i]) - np.min(mfcc[i])
                features[f'mfcc_{i}_skew'] = skew(mfcc[i])
                features[f'mfcc_{i}_kurtosis'] = kurtosis(mfcc[i])
                
                features[f'mfcc_delta_{i}_mean'] = np.mean(mfcc_delta[i])
                features[f'mfcc_delta_{i}_std'] = np.std(mfcc_delta[i])
                features[f'mfcc_delta2_{i}_mean'] = np.mean(mfcc_delta2[i])
                features[f'mfcc_delta2_{i}_std'] = np.std(mfcc_delta2[i])
        except:
            for i in range(self.n_mfcc):
                for suffix in ['_mean', '_std', '_max', '_min', '_median', '_range', '_skew', '_kurtosis']:
                    features[f'mfcc_{i}{suffix}'] = 0
                    features[f'mfcc_delta_{i}_mean'] = 0
                    features[f'mfcc_delta_{i}_std'] = 0
                    features[f'mfcc_delta2_{i}_mean'] = 0
                    features[f'mfcc_delta2_{i}_std'] = 0
        
        return features
    
    def _extract_spectral_features(self, y, sr):
        features = {}
        try:
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, 
                                                    n_fft=self.n_fft, hop_length=self.hop_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            for i in range(min(self.n_mels, mel_spec_db.shape[0])):
                features[f'mel_{i}_mean'] = np.mean(mel_spec_db[i])
                features[f'mel_{i}_std'] = np.std(mel_spec_db[i])
                features[f'mel_{i}_max'] = np.max(mel_spec_db[i])
                features[f'mel_{i}_min'] = np.min(mel_spec_db[i])
            
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, fmin=200.0)
            
            spectral_features = [
                ('spectral_centroid', spectral_centroids),
                ('spectral_rolloff', spectral_rolloff),
                ('spectral_bandwidth', spectral_bandwidth),
                ('spectral_flatness', spectral_flatness)
            ]
            
            for feature_name, feature_data in spectral_features:
                features[f'{feature_name}_mean'] = np.mean(feature_data)
                features[f'{feature_name}_std'] = np.std(feature_data)
                features[f'{feature_name}_max'] = np.max(feature_data)
                features[f'{feature_name}_min'] = np.min(feature_data)
                features[f'{feature_name}_median'] = np.median(feature_data)
                features[f'{feature_name}_skew'] = skew(feature_data)
                features[f'{feature_name}_kurtosis'] = kurtosis(feature_data)
            
            for i in range(spectral_contrast.shape[0]):
                features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
                features[f'spectral_contrast_{i}_std'] = np.std(spectral_contrast[i])
                features[f'spectral_contrast_{i}_max'] = np.max(spectral_contrast[i])
                features[f'spectral_contrast_{i}_min'] = np.min(spectral_contrast[i])
                
        except:
            for i in range(self.n_mels):
                for suffix in ['_mean', '_std', '_max', '_min']:
                    features[f'mel_{i}{suffix}'] = 0
            
            spectral_feature_names = ['spectral_centroid', 'spectral_rolloff', 
                                    'spectral_bandwidth', 'spectral_flatness']
            for feature_name in spectral_feature_names:
                for suffix in ['_mean', '_std', '_max', '_min', '_median', '_skew', '_kurtosis']:
                    features[f'{feature_name}{suffix}'] = 0
            
            for i in range(6):
                for suffix in ['_mean', '_std', '_max', '_min']:
                    features[f'spectral_contrast_{i}{suffix}'] = 0
        
        return features
    
    def _extract_prosodic_features(self, y, sr):
        features = {}
        try:
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
            f0_clean = f0[f0 > 0]
            
            if len(f0_clean) > 0:
                features.update({
                    'f0_mean': np.mean(f0_clean),
                    'f0_std': np.std(f0_clean),
                    'f0_max': np.max(f0_clean),
                    'f0_min': np.min(f0_clean),
                    'f0_range': np.max(f0_clean) - np.min(f0_clean),
                    'f0_median': np.median(f0_clean),
                    'f0_voiced_ratio': len(f0_clean) / len(f0),
                    'f0_skew': skew(f0_clean),
                    'f0_kurtosis': kurtosis(f0_clean)
                })
            else:
                f0_features = ['f0_mean', 'f0_std', 'f0_max', 'f0_min', 'f0_range', 
                              'f0_median', 'f0_voiced_ratio', 'f0_skew', 'f0_kurtosis']
                for feat in f0_features:
                    features[feat] = 0
        except:
            f0_features = ['f0_mean', 'f0_std', 'f0_max', 'f0_min', 'f0_range', 
                          'f0_median', 'f0_voiced_ratio', 'f0_skew', 'f0_kurtosis']
            for feat in f0_features:
                features[feat] = 0
        
        return features
    
    def _extract_energy_features(self, y, sr):
        features = {}
        try:
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
            
            for feature_name, feature_data in [('rms', rms), ('zcr', zcr)]:
                features[f'{feature_name}_mean'] = np.mean(feature_data)
                features[f'{feature_name}_std'] = np.std(feature_data)
                features[f'{feature_name}_max'] = np.max(feature_data)
                features[f'{feature_name}_min'] = np.min(feature_data)
                features[f'{feature_name}_median'] = np.median(feature_data)
                features[f'{feature_name}_skew'] = skew(feature_data)
                features[f'{feature_name}_kurtosis'] = kurtosis(feature_data)
                
        except:
            for feature_name in ['rms', 'zcr']:
                for suffix in ['_mean', '_std', '_max', '_min', '_median', '_skew', '_kurtosis']:
                    features[f'{feature_name}{suffix}'] = 0
        
        return features
    
    def _extract_chroma_features(self, y, sr):
        features = {}
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=self.n_chroma)
            for i in range(self.n_chroma):
                features[f'chroma_{i}_mean'] = np.mean(chroma[i])
                features[f'chroma_{i}_std'] = np.std(chroma[i])
                features[f'chroma_{i}_max'] = np.max(chroma[i])
                features[f'chroma_{i}_min'] = np.min(chroma[i])
        except:
            for i in range(self.n_chroma):
                for suffix in ['_mean', '_std', '_max', '_min']:
                    features[f'chroma_{i}{suffix}'] = 0
        
        return features
    
    def _extract_temporal_features(self, y, sr):
        features = {}
        features['duration'] = len(y) / sr
        features['amplitude_max'] = np.max(np.abs(y))
        features['amplitude_mean'] = np.mean(np.abs(y))
        features['amplitude_std'] = np.std(np.abs(y))
        features['amplitude_median'] = np.median(np.abs(y))
        features['energy_total'] = np.sum(y**2)
        features['energy_mean'] = np.mean(y**2)
        features['energy_std'] = np.std(y**2)
        return features
    
    def _extract_lpc_features(self, y):
        features = {}
        try:
            lpc_coeffs = librosa.lpc(y, order=self.lpc_order)
            for i, coeff in enumerate(lpc_coeffs[1:]):
                features[f'lpc_{i}'] = coeff
        except:
            for i in range(self.lpc_order):
                features[f'lpc_{i}'] = 0
        
        return features