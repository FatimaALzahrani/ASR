import numpy as np
import librosa
from scipy import stats


class AdvancedFeatureExtractor:
    def __init__(self, sr=16000):
        self.sr = sr
    
    def extract_features(self, audio, sr=None):
        if sr is None:
            sr = self.sr
        
        features = {}
        
        try:
            features['duration'] = len(audio) / sr
            features['rms_energy'] = float(np.sqrt(np.mean(audio**2)))
            features['energy_mean'] = float(np.mean(audio**2))
            features['energy_std'] = float(np.std(audio**2))
            
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
                features[f'mfcc_{i}_min'] = float(np.min(mfccs[i]))
                features[f'mfcc_{i}_max'] = float(np.max(mfccs[i]))
                
                features[f'delta_mfcc_{i}_mean'] = float(np.mean(delta_mfccs[i]))
                features[f'delta_mfcc_{i}_std'] = float(np.std(delta_mfccs[i]))
                
                features[f'delta2_mfcc_{i}_mean'] = float(np.mean(delta2_mfccs[i]))
                features[f'delta2_mfcc_{i}_std'] = float(np.std(delta2_mfccs[i]))
            
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)[0]
            features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
            features['spectral_contrast_std'] = float(np.std(spectral_contrast))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=10)
            for i in range(10):
                features[f'mel_{i}_mean'] = float(np.mean(mel_spec[i]))
                features[f'mel_{i}_std'] = float(np.std(mel_spec[i]))
            
            try:
                f0 = librosa.yin(audio, fmin=50, fmax=400)
                f0_clean = f0[f0 > 0]
                
                if len(f0_clean) > 0:
                    features['f0_mean'] = float(np.mean(f0_clean))
                    features['f0_std'] = float(np.std(f0_clean))
                    features['f0_min'] = float(np.min(f0_clean))
                    features['f0_max'] = float(np.max(f0_clean))
                    features['f0_range'] = features['f0_max'] - features['f0_min']
                    features['f0_skew'] = float(stats.skew(f0_clean))
                    features['f0_kurtosis'] = float(stats.kurtosis(f0_clean))
                else:
                    features.update({
                        'f0_mean': 0.0, 'f0_std': 0.0, 'f0_min': 0.0,
                        'f0_max': 0.0, 'f0_range': 0.0, 'f0_skew': 0.0,
                        'f0_kurtosis': 0.0
                    })
            except:
                features.update({
                    'f0_mean': 0.0, 'f0_std': 0.0, 'f0_min': 0.0,
                    'f0_max': 0.0, 'f0_range': 0.0, 'f0_skew': 0.0,
                    'f0_kurtosis': 0.0
                })
            
            try:
                features['energy_skew'] = float(stats.skew(audio**2))
                features['energy_kurtosis'] = float(stats.kurtosis(audio**2))
            except:
                features['energy_skew'] = 0.0
                features['energy_kurtosis'] = 0.0
            
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo)
            
            silence_threshold = 0.01
            silence_frames = np.sum(np.abs(audio) < silence_threshold)
            total_frames = len(audio)
            features['silence_ratio'] = silence_frames / total_frames
            features['speech_ratio'] = 1.0 - features['silence_ratio']
            
            signal_power = np.mean(audio**2)
            noise_power = np.var(audio - np.mean(audio))
            if noise_power > 0:
                features['snr_estimate'] = float(10 * np.log10(signal_power / noise_power))
            else:
                features['snr_estimate'] = 50.0
            
            if 'spectral_centroid_mean' in features and 'spectral_bandwidth_mean' in features:
                features['spectral_ratio'] = features['spectral_centroid_mean'] / (features['spectral_bandwidth_mean'] + 1e-8)
            
            if 'mfcc_0_mean' in features and 'mfcc_1_mean' in features:
                features['mfcc_ratio_0_1'] = features['mfcc_0_mean'] / (features['mfcc_1_mean'] + 1e-8)
            
            if 'rms_energy' in features and 'energy_mean' in features:
                features['energy_ratio'] = features['rms_energy'] / (features['energy_mean'] + 1e-8)
            
            if 'f0_mean' in features and 'f0_std' in features:
                features['f0_cv'] = features['f0_std'] / (features['f0_mean'] + 1e-8)
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return {}