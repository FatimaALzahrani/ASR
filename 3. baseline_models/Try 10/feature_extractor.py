import numpy as np
import librosa
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    def __init__(self):
        pass
    
    def extract_robust_features(self, audio_file, max_duration=3.0):
        try:
            y, sr = librosa.load(audio_file, sr=22050, duration=max_duration)
            
            if len(y) == 0:
                return None
                
            y = librosa.util.normalize(y)
            features = {}
            
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=256)
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                
                for i in range(20):
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
            except Exception as e:
                print(f"MFCC extraction error: {e}")
                for i in range(20):
                    for suffix in ['_mean', '_std', '_max', '_min', '_median', '_range', '_skew', '_kurtosis']:
                        features[f'mfcc_{i}{suffix}'] = 0
                        features[f'mfcc_delta_{i}_mean'] = 0
                        features[f'mfcc_delta_{i}_std'] = 0
                        features[f'mfcc_delta2_{i}_mean'] = 0
                        features[f'mfcc_delta2_{i}_std'] = 0
            
            try:
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=30, n_fft=1024, hop_length=256)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                for i in range(min(30, mel_spec_db.shape[0])):
                    features[f'mel_{i}_mean'] = np.mean(mel_spec_db[i])
                    features[f'mel_{i}_std'] = np.std(mel_spec_db[i])
                    features[f'mel_{i}_max'] = np.max(mel_spec_db[i])
                    features[f'mel_{i}_min'] = np.min(mel_spec_db[i])
            except Exception as e:
                print(f"Mel spectrogram error: {e}")
                for i in range(30):
                    features[f'mel_{i}_mean'] = 0
                    features[f'mel_{i}_std'] = 0
                    features[f'mel_{i}_max'] = 0
                    features[f'mel_{i}_min'] = 0
            
            try:
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
                    
            except Exception as e:
                print(f"Spectral features error: {e}")
                spectral_feature_names = ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'spectral_flatness']
                for feature_name in spectral_feature_names:
                    for suffix in ['_mean', '_std', '_max', '_min', '_median', '_skew', '_kurtosis']:
                        features[f'{feature_name}{suffix}'] = 0
                for i in range(6):
                    for suffix in ['_mean', '_std', '_max', '_min']:
                        features[f'spectral_contrast_{i}{suffix}'] = 0
            
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
            except Exception as e:
                print(f"F0 extraction error: {e}")
                f0_features = ['f0_mean', 'f0_std', 'f0_max', 'f0_min', 'f0_range', 
                              'f0_median', 'f0_voiced_ratio', 'f0_skew', 'f0_kurtosis']
                for feat in f0_features:
                    features[feat] = 0
            
            try:
                rms = librosa.feature.rms(y=y, hop_length=256)[0]
                zcr = librosa.feature.zero_crossing_rate(y, hop_length=256)[0]
                
                for feature_name, feature_data in [('rms', rms), ('zcr', zcr)]:
                    features[f'{feature_name}_mean'] = np.mean(feature_data)
                    features[f'{feature_name}_std'] = np.std(feature_data)
                    features[f'{feature_name}_max'] = np.max(feature_data)
                    features[f'{feature_name}_min'] = np.min(feature_data)
                    features[f'{feature_name}_median'] = np.median(feature_data)
                    features[f'{feature_name}_skew'] = skew(feature_data)
                    features[f'{feature_name}_kurtosis'] = kurtosis(feature_data)
                    
            except Exception as e:
                print(f"Energy features error: {e}")
                for feature_name in ['rms', 'zcr']:
                    for suffix in ['_mean', '_std', '_max', '_min', '_median', '_skew', '_kurtosis']:
                        features[f'{feature_name}{suffix}'] = 0
            
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
                for i in range(12):
                    features[f'chroma_{i}_mean'] = np.mean(chroma[i])
                    features[f'chroma_{i}_std'] = np.std(chroma[i])
                    features[f'chroma_{i}_max'] = np.max(chroma[i])
                    features[f'chroma_{i}_min'] = np.min(chroma[i])
            except Exception as e:
                print(f"Chroma features error: {e}")
                for i in range(12):
                    for suffix in ['_mean', '_std', '_max', '_min']:
                        features[f'chroma_{i}{suffix}'] = 0
            
            features.update({
                'duration': len(y) / sr,
                'amplitude_max': np.max(np.abs(y)),
                'amplitude_mean': np.mean(np.abs(y)),
                'amplitude_std': np.std(np.abs(y)),
                'amplitude_median': np.median(np.abs(y)),
                'energy_total': np.sum(y**2),
                'energy_mean': np.mean(y**2),
                'energy_std': np.std(y**2)
            })
            
            try:
                lpc_coeffs = librosa.lpc(y, order=12)
                for i, coeff in enumerate(lpc_coeffs[1:]):
                    features[f'lpc_{i}'] = coeff
            except Exception as e:
                print(f"LPC error: {e}")
                for i in range(12):
                    features[f'lpc_{i}'] = 0
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_file}: {e}")
            return None