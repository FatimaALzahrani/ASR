import numpy as np
import librosa
from utils import ultra_safe_stats
from config import MFCC_CONFIGS, F0_CONFIGS, QUALITY_MAPPING, MAX_DURATION, DEFAULT_SR

class FeatureExtractor:
    
    def extract_features(self, audio_file, speaker_profile):
        try:
            y, sr = librosa.load(audio_file, sr=DEFAULT_SR, duration=MAX_DURATION)
            
            if len(y) == 0:
                return None
            
            y = librosa.util.normalize(y)
            features = {}
            
            features.update(self._extract_mfcc_features(y, sr))
            features.update(self._extract_spectral_features(y, sr))
            features.update(self._extract_f0_features(y, sr))
            features.update(self._extract_temporal_features(y, sr))
            features.update(self._extract_harmonic_features(y, sr))
            features.update(self._extract_speaker_specific_features(y, sr, speaker_profile))
            
            clean_features = {}
            for key, value in features.items():
                if isinstance(value, (int, float, np.number)):
                    if np.isnan(value) or np.isinf(value):
                        clean_features[key] = 0.0
                    else:
                        clean_features[key] = float(value)
                else:
                    clean_features[key] = 0.0
            
            print(f"Extracted {len(clean_features)} features from {audio_file.name}")
            return clean_features
            
        except Exception as e:
            print(f"Feature extraction failed for {audio_file}: {e}")
            return None
    
    def _extract_mfcc_features(self, y, sr):
        features = {}
        
        try:
            for i, config in enumerate(MFCC_CONFIGS):
                try:
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, **config)
                    
                    if mfcc.shape[1] > 0:
                        for j in range(min(13, mfcc.shape[0])):
                            coeff_data = mfcc[j]
                            if len(coeff_data) > 0:
                                features.update(ultra_safe_stats(coeff_data, f'mfcc_{i}_{j}'))
                        
                        if mfcc.shape[1] > 10:
                            try:
                                width = min(9, max(3, mfcc.shape[1]//3))
                                mfcc_delta = librosa.feature.delta(mfcc, width=width)
                                for j in range(min(5, mfcc_delta.shape[0])):
                                    delta_data = mfcc_delta[j]
                                    if len(delta_data) > 0:
                                        features.update(ultra_safe_stats(delta_data, f'mfcc_delta_{i}_{j}'))
                            except Exception:
                                pass
                    
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        return features
    
    def _extract_spectral_features(self, y, sr):
        features = {}
        
        try:
            spectral_functions = [
                ('spectral_centroid', lambda: librosa.feature.spectral_centroid(y=y, sr=sr)[0]),
                ('spectral_rolloff', lambda: librosa.feature.spectral_rolloff(y=y, sr=sr)[0]),
                ('spectral_bandwidth', lambda: librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]),
                ('zcr', lambda: librosa.feature.zero_crossing_rate(y)[0]),
                ('rms', lambda: librosa.feature.rms(y=y)[0])
            ]
            
            for name, func in spectral_functions:
                try:
                    values = func()
                    if len(values) > 0:
                        features.update(ultra_safe_stats(values, name))
                except Exception:
                    for stat in ['mean', 'std', 'max', 'min']:
                        features[f'{name}_{stat}'] = 0.0
            
            try:
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=30)
                if mel_spec.shape[1] > 0:
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    for i in range(min(10, mel_spec_db.shape[0])):
                        mel_band = mel_spec_db[i]
                        if len(mel_band) > 0:
                            features.update(ultra_safe_stats(mel_band, f'mel_{i}'))
            except Exception:
                pass
                
        except Exception:
            pass
        
        return features
    
    def _extract_f0_features(self, y, sr):
        features = {}
        
        try:
            for i, config in enumerate(F0_CONFIGS):
                try:
                    f0 = librosa.yin(y, sr=sr, fmin=config['fmin'], fmax=config['fmax'])
                    f0_clean = f0[f0 > 0]
                    
                    if len(f0_clean) > 0:
                        features.update(ultra_safe_stats(f0_clean, f'f0_{i}'))
                        features[f'f0_{i}_voiced_ratio'] = len(f0_clean) / len(f0)
                        
                        if len(f0_clean) > 1:
                            f0_diff = np.diff(f0_clean)
                            if len(f0_diff) > 0 and np.mean(f0_clean) > 0:
                                features[f'f0_{i}_jitter'] = np.std(f0_diff) / np.mean(f0_clean)
                            else:
                                features[f'f0_{i}_jitter'] = 0.0
                        else:
                            features[f'f0_{i}_jitter'] = 0.0
                    else:
                        for stat in ['mean', 'std', 'max', 'min', 'voiced_ratio', 'jitter']:
                            features[f'f0_{i}_{stat}'] = 0.0
                            
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        return features
    
    def _extract_temporal_features(self, y, sr):
        features = {}
        
        try:
            duration = len(y) / sr
            features['duration'] = duration
            features['sample_rate'] = sr
            features['total_samples'] = len(y)
            
            try:
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
                if len(onset_frames) > 0:
                    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                    
                    features['onset_count'] = len(onset_times)
                    features['onset_rate'] = len(onset_times) / duration if duration > 0 else 0
                    
                    if len(onset_times) > 1:
                        onset_intervals = np.diff(onset_times)
                        features.update(ultra_safe_stats(onset_intervals, 'onset_interval'))
                else:
                    features['onset_count'] = 0
                    features['onset_rate'] = 0
                    for stat in ['mean', 'std']:
                        features[f'onset_interval_{stat}'] = 0.0
                        
            except Exception:
                features['onset_count'] = 0
                features['onset_rate'] = 0
            
            try:
                rms_values = librosa.feature.rms(y=y)[0]
                if len(rms_values) > 0:
                    silence_threshold = np.mean(rms_values) * 0.1
                    silence_frames = rms_values < silence_threshold
                    features['silence_ratio'] = np.sum(silence_frames) / len(silence_frames)
                else:
                    features['silence_ratio'] = 0.0
            except Exception:
                features['silence_ratio'] = 0.0
                
        except Exception:
            pass
        
        return features
    
    def _extract_harmonic_features(self, y, sr):
        features = {}
        
        try:
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                if chroma.shape[1] > 0:
                    for i in range(min(6, chroma.shape[0])):
                        chroma_data = chroma[i]
                        if len(chroma_data) > 0:
                            features.update(ultra_safe_stats(chroma_data, f'chroma_{i}'))
            except Exception:
                pass
            
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(y, margin=2.0)
                
                harmonic_energy = np.sum(y_harmonic**2)
                percussive_energy = np.sum(y_percussive**2)
                total_energy = harmonic_energy + percussive_energy
                
                if total_energy > 0:
                    features['harmonic_ratio'] = harmonic_energy / total_energy
                    features['percussive_ratio'] = percussive_energy / total_energy
                else:
                    features['harmonic_ratio'] = 0.5
                    features['percussive_ratio'] = 0.5
                    
            except Exception:
                features['harmonic_ratio'] = 0.5
                features['percussive_ratio'] = 0.5
                
        except Exception:
            pass
        
        return features
    
    def _extract_speaker_specific_features(self, y, sr, speaker_profile):
        features = {}
        
        try:
            speech_clarity = speaker_profile.get("clarity", 0.5)
            
            features['speaker_clarity'] = speech_clarity
            features['speaker_quality_numeric'] = QUALITY_MAPPING.get(
                speaker_profile.get("quality", "medium"), 0.5
            )
            
            try:
                spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
                if len(spectral_flatness) > 0:
                    features['ds_clarity_estimate'] = 1 - np.mean(spectral_flatness)
                    features['ds_clarity_variance'] = np.var(spectral_flatness)
                else:
                    features['ds_clarity_estimate'] = speech_clarity
                    features['ds_clarity_variance'] = 0.0
            except:
                features['ds_clarity_estimate'] = speech_clarity
                features['ds_clarity_variance'] = 0.0
            
            clarity_indicators = [
                features.get('ds_clarity_estimate', 0.5),
                1 - features.get('silence_ratio', 0.5),
                features.get('f0_0_voiced_ratio', 0.5) if 'f0_0_voiced_ratio' in features else 0.5,
                features['speaker_quality_numeric']
            ]
            
            features['ds_severity_index'] = np.mean(clarity_indicators)
            features['ds_speech_quality'] = 1 - features['ds_severity_index']
            
        except Exception:
            features['ds_severity_index'] = 0.5
            features['ds_speech_quality'] = 0.5
        
        return features
