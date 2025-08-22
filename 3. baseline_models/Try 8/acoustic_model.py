import os
import pandas as pd
import numpy as np
import librosa
import pickle
import re
from collections import Counter
import warnings
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional, Any
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AcousticModel:
    def __init__(self, sample_rate=22050, duration=3.0, random_state=42):
        self.sample_rate = sample_rate
        self.duration = duration
        self.random_state = random_state
        
        self.speaker_mapping = {
            'أحمد': list(range(0, 7)),
            'عاصم': list(range(7, 14)),
            'هيفاء': list(range(14, 21)),
            'أسيل': list(range(21, 29)),
            'وسام': list(range(29, 37))
        }
        
        self.number_to_speaker = {}
        for speaker, numbers in self.speaker_mapping.items():
            for num in numbers:
                self.number_to_speaker[num] = speaker
        
        self.feature_extractor = None
        self.speaker_models = {}
        self.global_models = {}
        self.ensemble_model = None
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        
        logger.info("Acoustic model initialized")
    
    def extract_speaker_from_filename(self, filename: str) -> str:
        name = os.path.splitext(filename)[0]
        number_match = re.search(r'(\d+)', name)
        if number_match:
            file_number = int(number_match.group(1))
            return self.number_to_speaker.get(file_number, 'unknown')
        return 'unknown'
    
    def load_data_from_folders(self, data_path: str) -> pd.DataFrame:
        data_records = []
        
        logger.info(f"Loading data from: {data_path}")
        
        for word_folder in os.listdir(data_path):
            word_path = os.path.join(data_path, word_folder)
            
            if not os.path.isdir(word_path):
                continue
                
            logger.info(f"Processing word: {word_folder}")
            
            for audio_file in os.listdir(word_path):
                if audio_file.endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    file_path = os.path.join(word_path, audio_file)
                    speaker = self.extract_speaker_from_filename(audio_file)
                    
                    if speaker != 'unknown':
                        data_records.append({
                            'file_path': file_path,
                            'word': word_folder,
                            'speaker': speaker,
                            'filename': audio_file
                        })
        
        df = pd.DataFrame(data_records)
        
        logger.info(f"Loaded {len(df)} samples, {df['word'].nunique()} words, {df['speaker'].nunique()} speakers")
        
        return df
    
    def advanced_audio_preprocessing(self, y: np.ndarray, sr: int) -> np.ndarray:
        try:
            y_preemphasized = np.append(y[0], y[1:] - 0.97 * y[:-1])
            y_denoised = self.spectral_gating_denoise(y_preemphasized, sr)
            y_normalized = librosa.util.normalize(y_denoised)
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
            logger.warning(f"Audio preprocessing failed: {str(e)}")
            return y
    
    def spectral_gating_denoise(self, y: np.ndarray, sr: int, alpha: float = 2.0, beta: float = 0.15) -> np.ndarray:
        try:
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
            
            mask = magnitude > (alpha * noise_floor)
            mask = mask.astype(float)
            
            mask = signal.medfilt(mask, kernel_size=(1, 3))
            
            magnitude_denoised = magnitude * (beta + (1 - beta) * mask)
            
            stft_denoised = magnitude_denoised * np.exp(1j * phase)
            y_denoised = librosa.istft(stft_denoised)
            
            return y_denoised
            
        except:
            return y
    
    def extract_comprehensive_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        features = {}
        
        try:
            features['duration'] = len(y) / sr
            features['rms_energy'] = float(np.sqrt(np.mean(y**2)))
            features['mean_amplitude'] = float(np.mean(np.abs(y)))
            features['std_amplitude'] = float(np.std(np.abs(y)))
            features['max_amplitude'] = float(np.max(np.abs(y)))
            features['min_amplitude'] = float(np.min(np.abs(y)))
            features['skewness'] = float(skew(y)) if len(y) > 1 else 0.0
            features['kurtosis'] = float(kurtosis(y)) if len(y) > 1 else 0.0
            
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            features['zcr_max'] = float(np.max(zcr))
            features['zcr_min'] = float(np.min(zcr))
            features['zcr_range'] = features['zcr_max'] - features['zcr_min']
            
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            features['spectral_centroid_max'] = float(np.max(spectral_centroids))
            features['spectral_centroid_min'] = float(np.min(spectral_centroids))
            features['spectral_centroid_range'] = features['spectral_centroid_max'] - features['spectral_centroid_min']
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            features['spectral_rolloff_max'] = float(np.max(spectral_rolloff))
            features['spectral_rolloff_min'] = float(np.min(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            features['spectral_bandwidth_max'] = float(np.max(spectral_bandwidth))
            features['spectral_bandwidth_min'] = float(np.min(spectral_bandwidth))
            
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            features['spectral_flatness_std'] = float(np.std(spectral_flatness))
            features['spectral_flatness_max'] = float(np.max(spectral_flatness))
            features['spectral_flatness_min'] = float(np.min(spectral_flatness))
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            for i in range(20):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
                features[f'mfcc_{i+1}_max'] = float(np.max(mfccs[i]))
                features[f'mfcc_{i+1}_min'] = float(np.min(mfccs[i]))
                features[f'mfcc_{i+1}_range'] = features[f'mfcc_{i+1}_max'] - features[f'mfcc_{i+1}_min']
            
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            for i in range(20):
                features[f'mfcc_delta_{i+1}_mean'] = float(np.mean(mfcc_delta[i]))
                features[f'mfcc_delta_{i+1}_std'] = float(np.std(mfcc_delta[i]))
                features[f'mfcc_delta2_{i+1}_mean'] = float(np.mean(mfcc_delta2[i]))
                features[f'mfcc_delta2_{i+1}_std'] = float(np.std(mfcc_delta2[i]))
            
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=26)
            mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            for i in range(26):
                features[f'mel_{i+1}_mean'] = float(np.mean(mel_db[i]))
                features[f'mel_{i+1}_std'] = float(np.std(mel_db[i]))
                features[f'mel_{i+1}_max'] = float(np.max(mel_db[i]))
                features[f'mel_{i+1}_min'] = float(np.min(mel_db[i]))
            
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            for i in range(12):
                features[f'chroma_{i+1}_mean'] = float(np.mean(chroma_stft[i]))
                features[f'chroma_{i+1}_std'] = float(np.std(chroma_stft[i]))
                features[f'chroma_{i+1}_max'] = float(np.max(chroma_stft[i]))
                features[f'chroma_{i+1}_min'] = float(np.min(chroma_stft[i]))
            
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            for i in range(7):
                features[f'spectral_contrast_{i+1}_mean'] = float(np.mean(spectral_contrast[i]))
                features[f'spectral_contrast_{i+1}_std'] = float(np.std(spectral_contrast[i]))
                features[f'spectral_contrast_{i+1}_max'] = float(np.max(spectral_contrast[i]))
                features[f'spectral_contrast_{i+1}_min'] = float(np.min(spectral_contrast[i]))
            
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            for i in range(6):
                features[f'tonnetz_{i+1}_mean'] = float(np.mean(tonnetz[i]))
                features[f'tonnetz_{i+1}_std'] = float(np.std(tonnetz[i]))
            
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)
            features['beat_density'] = len(beats) / (len(y) / sr)
            
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            features['harmonic_energy'] = float(np.sum(y_harmonic**2))
            features['percussive_energy'] = float(np.sum(y_percussive**2))
            features['harmonic_percussive_ratio'] = features['harmonic_energy'] / (features['percussive_energy'] + 1e-8)
            
            features['snr'] = self.estimate_snr(y)
            features['quality_score'] = self.calculate_quality_score(y, sr)
            features['silence_ratio'] = np.sum(np.abs(y) < 0.01) / len(y)
            features['clipping_ratio'] = np.sum(np.abs(y) > 0.99) / len(y)
            
            formants = self.estimate_formants(y, sr)
            for i, formant in enumerate(formants[:4]):
                features[f'formant_{i+1}'] = formant
            
            for key, value in features.items():
                if not np.isfinite(value):
                    features[key] = 0.0
            
            logger.debug(f"Extracted {len(features)} features")
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            features = {f'feature_{i}': 0.0 for i in range(300)}
        
        return features
    
    def estimate_snr(self, signal: np.ndarray) -> float:
        try:
            signal_power = np.mean(signal**2)
            noise_power = signal_power * 0.01
            if noise_power == 0:
                return 40.0
            snr = 10 * np.log10(signal_power / noise_power)
            return max(snr, -10.0)
        except:
            return 20.0
    
    def calculate_quality_score(self, y: np.ndarray, sr: int) -> float:
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
    
    def estimate_formants(self, y: np.ndarray, sr: int) -> List[float]:
        try:
            fft = np.fft.fft(y)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(fft), 1/sr)[:len(fft)//2]
            
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(magnitude, height=np.max(magnitude)*0.1)
            
            peak_freqs = freqs[peaks]
            
            peak_magnitudes = magnitude[peaks]
            sorted_indices = np.argsort(peak_magnitudes)[::-1]
            
            formants = []
            for i in sorted_indices[:4]:
                formants.append(float(peak_freqs[i]))
            
            while len(formants) < 4:
                formants.append(0.0)
            
            return formants
            
        except:
            return [0.0, 0.0, 0.0, 0.0]
    
    def process_audio_file(self, file_path: str) -> Optional[Dict[str, float]]:
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            y_processed = self.advanced_audio_preprocessing(y, sr)
            features = self.extract_comprehensive_features(y_processed, sr)
            return features
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            return None
    
    def train_acoustic_models(self, data_path: str, min_samples_per_word: int = 3) -> Dict[str, Any]:
        logger.info("Starting acoustic model training")
        
        df = self.load_data_from_folders(data_path)
        
        word_counts = df['word'].value_counts()
        valid_words = word_counts[word_counts >= min_samples_per_word].index
        df_filtered = df[df['word'].isin(valid_words)]
        
        logger.info(f"Filtered dataset: {len(df_filtered)} samples, {len(valid_words)} words")
        
        logger.info("Processing audio files")
        processed_records = []
        
        for idx, row in df_filtered.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing {idx+1}/{len(df_filtered)}")
            
            features = self.process_audio_file(row['file_path'])
            
            if features is not None:
                record = {
                    'file_path': row['file_path'],
                    'word': row['word'],
                    'speaker': row['speaker'],
                    'filename': row['filename']
                }
                record.update(features)
                processed_records.append(record)
        
        logger.info(f"Successfully processed {len(processed_records)} files")
        
        features_df = pd.DataFrame(processed_records)
        
        self.feature_columns = [col for col in features_df.columns 
                               if col not in ['file_path', 'word', 'speaker', 'filename']]
        
        if 'quality_score' in features_df.columns:
            quality_threshold = 0.3
            high_quality_mask = features_df['quality_score'] >= quality_threshold
            features_df = features_df[high_quality_mask]
            logger.info(f"Removed {(~high_quality_mask).sum()} low-quality samples")
        
        self._train_speaker_specific_models(features_df)
        self._train_global_models(features_df)
        self._train_ensemble_model(features_df)
        
        self.label_encoder.fit(features_df['word'])
        
        logger.info("Acoustic model training completed")
        
        return {
            'speaker_models': self.speaker_models,
            'global_models': self.global_models,
            'ensemble_model': self.ensemble_model,
            'feature_columns': self.feature_columns,
            'label_encoder': self.label_encoder
        }
    
    def _train_speaker_specific_models(self, df: pd.DataFrame):
        logger.info("Training speaker-specific models")
        
        for speaker in df['speaker'].unique():
            logger.info(f"Training models for speaker: {speaker}")
            
            speaker_data = df[df['speaker'] == speaker]
            
            if len(speaker_data) < 10:
                logger.warning(f"Insufficient data for {speaker}, skipping")
                continue
            
            X = speaker_data[self.feature_columns].fillna(0)
            y = speaker_data['word']
            
            feature_std = X.std()
            non_constant_features = feature_std[feature_std > 1e-8].index
            X = X[non_constant_features]
            
            if len(X.columns) < 10:
                logger.warning(f"Too few features for {speaker}, skipping")
                continue
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=self.random_state,
                    stratify=y if len(y.unique()) > 1 else None
                )
            except:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=self.random_state
                )
            
            n_features = min(50, len(X.columns))
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    random_state=self.random_state, n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=150, max_depth=8, learning_rate=0.1,
                    random_state=self.random_state
                ),
                'svm': SVC(
                    C=1.0, gamma='scale', probability=True,
                    random_state=self.random_state
                ),
                'logistic_regression': LogisticRegression(
                    max_iter=1000, random_state=self.random_state
                )
            }
            
            speaker_models = {}
            
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    speaker_models[name] = {
                        'model': model,
                        'scaler': scaler,
                        'selector': selector,
                        'feature_columns': X.columns.tolist(),
                        'accuracy': accuracy
                    }
                    
                    logger.info(f"  {name}: {accuracy:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name} for {speaker}: {str(e)}")
            
            self.speaker_models[speaker] = speaker_models
    
    def _train_global_models(self, df: pd.DataFrame):
        logger.info("Training global models")
        
        X = df[self.feature_columns].fillna(0)
        y = df['word']
        
        feature_std = X.std()
        non_constant_features = feature_std[feature_std > 1e-8].index
        X = X[non_constant_features]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        n_features = min(100, len(X.columns))
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_split=5,
                random_state=self.random_state, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                random_state=self.random_state
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=300, max_depth=20, min_samples_split=5,
                random_state=self.random_state, n_jobs=-1
            )
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.global_models[name] = {
                    'model': model,
                    'scaler': scaler,
                    'selector': selector,
                    'feature_columns': X.columns.tolist(),
                    'accuracy': accuracy
                }
                
                logger.info(f"Global {name}: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error training global {name}: {str(e)}")
    
    def _train_ensemble_model(self, df: pd.DataFrame):
        logger.info("Training ensemble model")
        
        if self.global_models:
            best_model_name = max(self.global_models.keys(), 
                                key=lambda k: self.global_models[k]['accuracy'])
            self.ensemble_model = self.global_models[best_model_name]
            logger.info(f"Ensemble model: {best_model_name} with accuracy {self.ensemble_model['accuracy']:.4f}")
    
    def predict(self, audio_file_path: str, speaker: Optional[str] = None) -> Tuple[str, float]:
        try:
            features = self.process_audio_file(audio_file_path)
            if features is None:
                return "unknown", 0.0
            
            feature_df = pd.DataFrame([features])
            
            if speaker and speaker in self.speaker_models:
                models = self.speaker_models[speaker]
                best_model_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
                model_data = models[best_model_name]
            else:
                if not self.ensemble_model:
                    return "unknown", 0.0
                model_data = self.ensemble_model
            
            X = feature_df[model_data['feature_columns']].fillna(0)
            X_selected = model_data['selector'].transform(X)
            X_scaled = model_data['scaler'].transform(X_selected)
            
            prediction = model_data['model'].predict(X_scaled)[0]
            
            try:
                probabilities = model_data['model'].predict_proba(X_scaled)[0]
                confidence = float(np.max(probabilities))
            except:
                confidence = 0.5
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return "unknown", 0.0
    
    def save_model(self, model_path: str):
        model_data = {
            'speaker_models': self.speaker_models,
            'global_models': self.global_models,
            'ensemble_model': self.ensemble_model,
            'feature_columns': self.feature_columns,
            'label_encoder': self.label_encoder,
            'speaker_mapping': self.speaker_mapping,
            'sample_rate': self.sample_rate,
            'duration': self.duration
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Acoustic model saved to: {model_path}")
    
    def load_model(self, model_path: str):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.speaker_models = model_data['speaker_models']
        self.global_models = model_data['global_models']
        self.ensemble_model = model_data['ensemble_model']
        self.feature_columns = model_data['feature_columns']
        self.label_encoder = model_data['label_encoder']
        self.speaker_mapping = model_data['speaker_mapping']
        self.sample_rate = model_data['sample_rate']
        self.duration = model_data['duration']
        
        logger.info(f"Acoustic model loaded from: {model_path}")
