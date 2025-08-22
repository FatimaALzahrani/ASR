from config import *

class AudioProcessor:
    def __init__(self, sample_rate=Config.SAMPLE_RATE, duration=Config.DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
    
    def advanced_audio_preprocessing(self, y, sr):
        try:
            y_preemphasized = np.append(y[0], y[1:] - 0.97 * y[:-1])
            y_normalized = librosa.util.normalize(y_preemphasized)
            y_trimmed, _ = librosa.effects.trim(y_normalized, top_db=15)
            
            target_length = int(self.duration * sr)
            if len(y_trimmed) > target_length:
                start = (len(y_trimmed) - target_length) // 2
                y_final = y_trimmed[start:start + target_length]
            else:
                pad_length = target_length - len(y_trimmed)
                y_final = np.pad(y_trimmed, (0, pad_length), mode='constant')
            
            return y_final
            
        except Exception as e:
            print(f"Audio preprocessing failed: {str(e)}")
            return y
    
    def extract_robust_features(self, y, sr):
        features = {}
        
        try:
            features['duration'] = len(y) / sr
            features['rms_energy'] = float(np.sqrt(np.mean(y**2)))
            features['mean_amplitude'] = float(np.mean(np.abs(y)))
            features['std_amplitude'] = float(np.std(np.abs(y)))
            features['max_amplitude'] = float(np.max(np.abs(y)))
            features['skewness'] = float(skew(y)) if len(y) > 1 else 0.0
            features['kurtosis'] = float(kurtosis(y)) if len(y) > 1 else 0.0
            
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            features['zcr_max'] = float(np.max(zcr))
            features['zcr_min'] = float(np.min(zcr))
            
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            features['spectral_centroid_max'] = float(np.max(spectral_centroids))
            features['spectral_centroid_min'] = float(np.min(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            features['spectral_flatness_std'] = float(np.std(spectral_flatness))
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=Config.N_MFCC)
            for i in range(Config.N_MFCC):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
                features[f'mfcc_{i+1}_max'] = float(np.max(mfccs[i]))
                features[f'mfcc_{i+1}_min'] = float(np.min(mfccs[i]))
            
            mfcc_delta = librosa.feature.delta(mfccs)
            for i in range(Config.N_MFCC):
                features[f'mfcc_delta_{i+1}_mean'] = float(np.mean(mfcc_delta[i]))
                features[f'mfcc_delta_{i+1}_std'] = float(np.std(mfcc_delta[i]))
            
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=Config.N_MEL)
            mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            for i in range(Config.N_MEL):
                features[f'mel_{i+1}_mean'] = float(np.mean(mel_db[i]))
                features[f'mel_{i+1}_std'] = float(np.std(mel_db[i]))
                features[f'mel_{i+1}_max'] = float(np.max(mel_db[i]))
                features[f'mel_{i+1}_min'] = float(np.min(mel_db[i]))
            
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            for i in range(Config.N_CHROMA):
                features[f'chroma_{i+1}_mean'] = float(np.mean(chroma_stft[i]))
                features[f'chroma_{i+1}_std'] = float(np.std(chroma_stft[i]))
            
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            for i in range(Config.N_SPECTRAL_CONTRAST):
                features[f'spectral_contrast_{i+1}_mean'] = float(np.mean(spectral_contrast[i]))
                features[f'spectral_contrast_{i+1}_std'] = float(np.std(spectral_contrast[i]))
            
            features['snr'] = self.estimate_snr(y)
            features['quality_score'] = self.calculate_quality_score(y, sr)
            
            for key, value in features.items():
                if not np.isfinite(value):
                    features[key] = 0.0
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            default_features = {}
            
            basic_names = ['duration', 'rms_energy', 'mean_amplitude', 'std_amplitude', 'max_amplitude',
                          'skewness', 'kurtosis', 'zcr_mean', 'zcr_std', 'zcr_max', 'zcr_min']
            
            spectral_names = ['spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid_max', 'spectral_centroid_min',
                             'spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                             'spectral_flatness_mean', 'spectral_flatness_std']
            
            mfcc_names = []
            for i in range(Config.N_MFCC):
                mfcc_names.extend([f'mfcc_{i+1}_mean', f'mfcc_{i+1}_std', f'mfcc_{i+1}_max', f'mfcc_{i+1}_min'])
            
            delta_names = []
            for i in range(Config.N_MFCC):
                delta_names.extend([f'mfcc_delta_{i+1}_mean', f'mfcc_delta_{i+1}_std'])
            
            mel_names = []
            for i in range(Config.N_MEL):
                mel_names.extend([f'mel_{i+1}_mean', f'mel_{i+1}_std', f'mel_{i+1}_max', f'mel_{i+1}_min'])
            
            chroma_names = []
            for i in range(Config.N_CHROMA):
                chroma_names.extend([f'chroma_{i+1}_mean', f'chroma_{i+1}_std'])
            
            contrast_names = []
            for i in range(Config.N_SPECTRAL_CONTRAST):
                contrast_names.extend([f'spectral_contrast_{i+1}_mean', f'spectral_contrast_{i+1}_std'])
            
            quality_names = ['snr', 'quality_score']
            
            all_feature_names = basic_names + spectral_names + mfcc_names + delta_names + mel_names + chroma_names + contrast_names + quality_names
            
            features = {name: 0.0 for name in all_feature_names}
        
        return features
    
    def estimate_snr(self, signal):
        try:
            signal_power = np.mean(signal**2)
            noise_power = signal_power * 0.01
            if noise_power == 0:
                return 40.0
            snr = 10 * np.log10(signal_power / noise_power)
            return max(snr, -10.0)
        except:
            return 20.0
    
    def calculate_quality_score(self, y, sr):
        try:
            snr = self.estimate_snr(y)
            snr_score = max(0, min((snr + 10) / 50, 1.0))
            
            clipping_rate = np.sum(np.abs(y) > 0.99) / len(y)
            clipping_score = 1.0 - clipping_rate
            
            silence_rate = np.sum(np.abs(y) < 0.01) / len(y)
            silence_score = 1.0 - min(silence_rate, 0.8) / 0.8
            
            quality = (snr_score * 0.5 + clipping_score * 0.3 + silence_score * 0.2)
            return max(0.0, min(quality, 1.0))
        except:
            return 0.5
    
    def process_audio_file(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            y_processed = self.advanced_audio_preprocessing(y, sr)
            features = self.extract_robust_features(y_processed, sr)
            return features
        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")
            return None