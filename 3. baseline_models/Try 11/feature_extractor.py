import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import skew, kurtosis
from config import Config


class RobustFeatureExtractor:
    
    def __init__(self, sample_rate=None):
        self.sr = sample_rate or Config.SAMPLE_RATE
        self.features_extracted = 0
        self.failed_extractions = 0
        
    def extract_comprehensive_features(self, audio_file, word=None, speaker=None):
        
        try:
            y, sr = librosa.load(audio_file, sr=self.sr, duration=Config.AUDIO_DURATION)
            
            if len(y) == 0:
                print(f"Empty file: {audio_file}")
                return self._get_zero_features()
            
            y = librosa.util.normalize(y)
            
            features = {}
            
            features.update(self._extract_basic_features(y, sr))
            features.update(self._extract_mfcc_features(y, sr))
            features.update(self._extract_spectral_features(y, sr))
            features.update(self._extract_energy_rhythm_features(y, sr))
            features.update(self._extract_f0_features(y, sr))
            
            if word and speaker:
                features.update(self._extract_metadata_features(word, speaker))
            
            features.update(self._extract_statistical_features(y))
            
            self.features_extracted += 1
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_file}: {e}")
            self.failed_extractions += 1
            return self._get_zero_features()
    
    def _extract_basic_features(self, y, sr):
        features = {}
        
        try:
            features['duration'] = len(y) / sr
            
            features['signal_mean'] = np.mean(y)
            features['signal_std'] = np.std(y)
            features['signal_max'] = np.max(y)
            features['signal_min'] = np.min(y)
            features['signal_range'] = features['signal_max'] - features['signal_min']
            
            features['energy_total'] = np.sum(y**2)
            features['energy_mean'] = np.mean(y**2)
            features['energy_std'] = np.std(y**2)
            
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
        except Exception as e:
            print(f"Error in basic features: {e}")
            for key in ['duration', 'signal_mean', 'signal_std', 'signal_max', 'signal_min', 
                       'signal_range', 'energy_total', 'energy_mean', 'energy_std', 
                       'zcr_mean', 'zcr_std']:
                features[key] = 0.0
        
        return features
    
    def _extract_mfcc_features(self, y, sr):
        features = {}
        
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=Config.MFCC_COEFFICIENTS, n_fft=2048, hop_length=512)
            
            for i in range(Config.MFCC_COEFFICIENTS):
                coeff = mfccs[i]
                features[f'mfcc_{i}_mean'] = np.mean(coeff)
                features[f'mfcc_{i}_std'] = np.std(coeff)
                features[f'mfcc_{i}_max'] = np.max(coeff)
                features[f'mfcc_{i}_min'] = np.min(coeff)
                features[f'mfcc_{i}_median'] = np.median(coeff)
                features[f'mfcc_{i}_range'] = np.max(coeff) - np.min(coeff)
            
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            for i in range(5):
                features[f'mfcc_delta_{i}_mean'] = np.mean(mfcc_delta[i])
                features[f'mfcc_delta_{i}_std'] = np.std(mfcc_delta[i])
                features[f'mfcc_delta2_{i}_mean'] = np.mean(mfcc_delta2[i])
                features[f'mfcc_delta2_{i}_std'] = np.std(mfcc_delta2[i])
                
        except Exception as e:
            print(f"Error in MFCC: {e}")
            for i in range(13):
                features[f'mfcc_{i}_mean'] = 0.0
                features[f'mfcc_{i}_std'] = 0.0
                features[f'mfcc_{i}_max'] = 0.0
                features[f'mfcc_{i}_min'] = 0.0
                features[f'mfcc_{i}_median'] = 0.0
                features[f'mfcc_{i}_range'] = 0.0
            
            for i in range(5):
                features[f'mfcc_delta_{i}_mean'] = 0.0
                features[f'mfcc_delta_{i}_std'] = 0.0
                features[f'mfcc_delta2_{i}_mean'] = 0.0
                features[f'mfcc_delta2_{i}_std'] = 0.0
        
        return features
    
    def _extract_spectral_features(self, y, sr):
        features = {}
        
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features['spectral_flatness_mean'] = np.mean(spectral_flatness)
            features['spectral_flatness_std'] = np.std(spectral_flatness)
            
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=Config.MEL_BANDS)
            mel_spec_db = librosa.power_to_db(mel_spec)
            
            for i in range(min(10, mel_spec_db.shape[0])):
                features[f'mel_{i}_mean'] = np.mean(mel_spec_db[i])
                features[f'mel_{i}_std'] = np.std(mel_spec_db[i])
                
        except Exception as e:
            print(f"Error in spectral features: {e}")
            spectral_features = [
                'spectral_centroid_mean', 'spectral_centroid_std',
                'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                'spectral_rolloff_mean', 'spectral_rolloff_std',
                'spectral_flatness_mean', 'spectral_flatness_std'
            ]
            
            for feat in spectral_features:
                features[feat] = 0.0
                
            for i in range(10):
                features[f'mel_{i}_mean'] = 0.0
                features[f'mel_{i}_std'] = 0.0
        
        return features
    
    def _extract_energy_rhythm_features(self, y, sr):
        features = {}
        
        try:
            rms = librosa.feature.rms(y=y, hop_length=512)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            features['rms_max'] = np.max(rms)
            features['rms_min'] = np.min(rms)
            
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            features['onset_count'] = len(onset_frames)
            features['onset_density'] = len(onset_frames) / (len(y) / sr)
            
            if len(onset_frames) > 1:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                onset_intervals = np.diff(onset_times)
                features['onset_interval_mean'] = np.mean(onset_intervals)
                features['onset_interval_std'] = np.std(onset_intervals)
            else:
                features['onset_interval_mean'] = 0.0
                features['onset_interval_std'] = 0.0
            
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = tempo
                features['beat_count'] = len(beats)
            except:
                features['tempo'] = 0.0
                features['beat_count'] = 0
                
        except Exception as e:
            print(f"Error in energy/rhythm features: {e}")
            energy_rhythm_features = [
                'rms_mean', 'rms_std', 'rms_max', 'rms_min',
                'onset_count', 'onset_density', 'onset_interval_mean', 
                'onset_interval_std', 'tempo', 'beat_count'
            ]
            
            for feat in energy_rhythm_features:
                features[feat] = 0.0
        
        return features
    
    def _extract_f0_features(self, y, sr):
        features = {}
        
        try:
            f0 = librosa.yin(y, fmin=Config.F0_MIN, fmax=Config.F0_MAX, sr=sr)
            f0_clean = f0[f0 > 0]
            
            if len(f0_clean) > 0:
                features['f0_mean'] = np.mean(f0_clean)
                features['f0_std'] = np.std(f0_clean)
                features['f0_max'] = np.max(f0_clean)
                features['f0_min'] = np.min(f0_clean)
                features['f0_range'] = features['f0_max'] - features['f0_min']
                features['f0_median'] = np.median(f0_clean)
                features['voicing_ratio'] = len(f0_clean) / len(f0)
                
                if len(f0_clean) > 1:
                    f0_diff = np.abs(np.diff(f0_clean))
                    features['f0_jitter'] = np.mean(f0_diff) / np.mean(f0_clean)
                else:
                    features['f0_jitter'] = 0.0
            else:
                f0_features = ['f0_mean', 'f0_std', 'f0_max', 'f0_min', 'f0_range', 
                              'f0_median', 'voicing_ratio', 'f0_jitter']
                for feat in f0_features:
                    features[feat] = 0.0
                    
        except Exception as e:
            print(f"Error in F0: {e}")
            f0_features = ['f0_mean', 'f0_std', 'f0_max', 'f0_min', 'f0_range', 
                          'f0_median', 'voicing_ratio', 'f0_jitter']
            for feat in f0_features:
                features[feat] = 0.0
        
        return features
    
    def _extract_metadata_features(self, word, speaker):
        features = {}
        
        try:
            difficult_chars = Config.DIFFICULT_ARABIC_CHARS
            fricative_chars = Config.FRICATIVE_ARABIC_CHARS
            
            features['word_length'] = len(word)
            features['difficult_chars_count'] = sum(1 for c in word if c in difficult_chars)
            features['fricative_chars_count'] = sum(1 for c in word if c in fricative_chars)
            features['word_unique_chars'] = len(set(word))
            features['word_repetition_ratio'] = (len(word) - len(set(word))) / max(1, len(word))
            
            speaker_encoding = {
                'أحمد': 1, 'عاصم': 2, 'هيفاء': 3, 'أسيل': 4, 'وسام': 5
            }
            features['speaker_encoded'] = speaker_encoding.get(speaker, 0)
            
        except Exception as e:
            print(f"Error in metadata features: {e}")
            metadata_features = [
                'word_length', 'difficult_chars_count', 'fricative_chars_count',
                'word_unique_chars', 'word_repetition_ratio', 'speaker_encoded'
            ]
            for feat in metadata_features:
                features[feat] = 0.0
        
        return features
    
    def _extract_statistical_features(self, y):
        features = {}
        
        try:
            features['signal_skew'] = float(skew(y)) if len(y) > 1 else 0.0
            features['signal_kurtosis'] = float(kurtosis(y)) if len(y) > 1 else 0.0
            
            features['signal_variation'] = np.std(y) / np.mean(np.abs(y)) if np.mean(np.abs(y)) > 0 else 0.0
            
            hist, _ = np.histogram(y, bins=20)
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]
            features['signal_entropy'] = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
            
        except Exception as e:
            print(f"Error in statistical features: {e}")
            features['signal_skew'] = 0.0
            features['signal_kurtosis'] = 0.0
            features['signal_variation'] = 0.0
            features['signal_entropy'] = 0.0
        
        return features
    
    def _get_zero_features(self):
        features = {}
        
        basic_features = [
            'duration', 'signal_mean', 'signal_std', 'signal_max', 'signal_min', 
            'signal_range', 'energy_total', 'energy_mean', 'energy_std', 
            'zcr_mean', 'zcr_std'
        ]
        
        mfcc_features = []
        for i in range(13):
            mfcc_features.extend([
                f'mfcc_{i}_mean', f'mfcc_{i}_std', f'mfcc_{i}_max', 
                f'mfcc_{i}_min', f'mfcc_{i}_median', f'mfcc_{i}_range'
            ])
        
        for i in range(5):
            mfcc_features.extend([
                f'mfcc_delta_{i}_mean', f'mfcc_delta_{i}_std',
                f'mfcc_delta2_{i}_mean', f'mfcc_delta2_{i}_std'
            ])
        
        spectral_features = [
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_flatness_mean', 'spectral_flatness_std'
        ]
        
        for i in range(10):
            spectral_features.extend([f'mel_{i}_mean', f'mel_{i}_std'])
        
        other_features = [
            'rms_mean', 'rms_std', 'rms_max', 'rms_min',
            'onset_count', 'onset_density', 'onset_interval_mean', 
            'onset_interval_std', 'tempo', 'beat_count',
            'f0_mean', 'f0_std', 'f0_max', 'f0_min', 'f0_range', 
            'f0_median', 'voicing_ratio', 'f0_jitter',
            'word_length', 'difficult_chars_count', 'fricative_chars_count',
            'word_unique_chars', 'word_repetition_ratio', 'speaker_encoded',
            'signal_skew', 'signal_kurtosis', 'signal_variation', 'signal_entropy'
        ]
        
        all_features = basic_features + mfcc_features + spectral_features + other_features
        
        for feat in all_features:
            features[feat] = 0.0
        
        return features