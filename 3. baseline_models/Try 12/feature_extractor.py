import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

class ConservativeFeatureExtractor:
    
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate
        self.features_extracted = 0
        self.failed_extractions = 0
        
    def extract_essential_features(self, audio_file, word=None, speaker=None):
        
        try:
            y, sr = librosa.load(audio_file, sr=self.sr, duration=4.0)
            
            if len(y) == 0:
                return self._get_zero_features()
            
            y = librosa.util.normalize(y)
            features = {}
            
            features.update(self._extract_core_mfcc(y, sr))
            features.update(self._extract_core_spectral(y, sr))
            features.update(self._extract_core_f0(y, sr))
            features.update(self._extract_core_energy(y, sr))
            
            if word and speaker:
                features.update(self._extract_metadata_simple(word, speaker))
            
            self.features_extracted += 1
            return features
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            self.failed_extractions += 1
            return self._get_zero_features()
    
    def _extract_core_mfcc(self, y, sr):
        features = {}
        
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, n_fft=2048, hop_length=512)
            
            for i in range(8):
                coeff = mfccs[i]
                features[f'mfcc_{i}_mean'] = np.mean(coeff)
                features[f'mfcc_{i}_std'] = np.std(coeff)
                features[f'mfcc_{i}_max'] = np.max(coeff)
                features[f'mfcc_{i}_min'] = np.min(coeff)
            
            mfcc_delta = librosa.feature.delta(mfccs)
            for i in range(3):
                features[f'mfcc_delta_{i}_mean'] = np.mean(mfcc_delta[i])
                features[f'mfcc_delta_{i}_std'] = np.std(mfcc_delta[i])
                
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            for i in range(8):
                features[f'mfcc_{i}_mean'] = 0.0
                features[f'mfcc_{i}_std'] = 0.0
                features[f'mfcc_{i}_max'] = 0.0
                features[f'mfcc_{i}_min'] = 0.0
            for i in range(3):
                features[f'mfcc_delta_{i}_mean'] = 0.0
                features[f'mfcc_delta_{i}_std'] = 0.0
        
        return features
    
    def _extract_core_spectral(self, y, sr):
        features = {}
        
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10)
            mel_spec_db = librosa.power_to_db(mel_spec)
            
            for i in range(5):
                features[f'mel_{i}_mean'] = np.mean(mel_spec_db[i])
                
        except Exception as e:
            print(f"Spectral features error: {e}")
            spectral_features = [
                'spectral_centroid_mean', 'spectral_centroid_std',
                'spectral_bandwidth_mean', 'spectral_rolloff_mean'
            ]
            for feat in spectral_features:
                features[feat] = 0.0
            for i in range(5):
                features[f'mel_{i}_mean'] = 0.0
        
        return features
    
    def _extract_core_f0(self, y, sr):
        features = {}
        
        try:
            f0 = librosa.yin(y, fmin=50, fmax=300, sr=sr)
            f0_clean = f0[f0 > 0]
            
            if len(f0_clean) > 0:
                features['f0_mean'] = np.mean(f0_clean)
                features['f0_std'] = np.std(f0_clean)
                features['f0_range'] = np.max(f0_clean) - np.min(f0_clean)
                features['voicing_ratio'] = len(f0_clean) / len(f0)
            else:
                features['f0_mean'] = 0.0
                features['f0_std'] = 0.0
                features['f0_range'] = 0.0
                features['voicing_ratio'] = 0.0
                
        except Exception as e:
            print(f"F0 extraction error: {e}")
            features['f0_mean'] = 0.0
            features['f0_std'] = 0.0
            features['f0_range'] = 0.0
            features['voicing_ratio'] = 0.0
        
        return features
    
    def _extract_core_energy(self, y, sr):
        features = {}
        
        try:
            rms = librosa.feature.rms(y=y, hop_length=512)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            
            features['signal_mean'] = np.mean(y)
            features['signal_std'] = np.std(y)
            features['signal_max'] = np.max(y)
            features['duration'] = len(y) / sr
            
        except Exception as e:
            print(f"Energy features error: {e}")
            energy_features = ['rms_mean', 'rms_std', 'zcr_mean', 
                             'signal_mean', 'signal_std', 'signal_max', 'duration']
            for feat in energy_features:
                features[feat] = 0.0
        
        return features
    
    def _extract_metadata_simple(self, word, speaker):
        features = {}
        
        try:
            difficult_chars = ['خ', 'غ', 'ق', 'ض', 'ظ', 'ث', 'ذ', 'ص', 'ز']
            
            features['word_length'] = len(word)
            features['difficult_chars_count'] = sum(1 for c in word if c in difficult_chars)
            features['word_difficulty_ratio'] = features['difficult_chars_count'] / max(1, len(word))
            
            speaker_encoding = {
                'أحمد': 1, 'عاصم': 2, 'هيفاء': 3, 'أسيل': 4, 'وسام': 5
            }
            features['speaker_encoded'] = speaker_encoding.get(speaker, 0)
            
        except Exception as e:
            print(f"Metadata extraction error: {e}")
            features['word_length'] = 0.0
            features['difficult_chars_count'] = 0.0
            features['word_difficulty_ratio'] = 0.0
            features['speaker_encoded'] = 0.0
        
        return features
    
    def _get_zero_features(self):
        features = {}
        
        for i in range(8):
            features[f'mfcc_{i}_mean'] = 0.0
            features[f'mfcc_{i}_std'] = 0.0
            features[f'mfcc_{i}_max'] = 0.0
            features[f'mfcc_{i}_min'] = 0.0
        
        for i in range(3):
            features[f'mfcc_delta_{i}_mean'] = 0.0
            features[f'mfcc_delta_{i}_std'] = 0.0
        
        spectral_features = [
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_rolloff_mean'
        ]
        for feat in spectral_features:
            features[feat] = 0.0
        
        for i in range(5):
            features[f'mel_{i}_mean'] = 0.0
        
        other_features = [
            'f0_mean', 'f0_std', 'f0_range', 'voicing_ratio',
            'rms_mean', 'rms_std', 'zcr_mean',
            'signal_mean', 'signal_std', 'signal_max', 'duration',
            'word_length', 'difficult_chars_count', 'word_difficulty_ratio', 'speaker_encoded'
        ]
        for feat in other_features:
            features[feat] = 0.0
        
        return features