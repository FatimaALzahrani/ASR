import numpy as np
import librosa
from scipy.stats import skew, kurtosis
from audio_preprocessor import AudioPreprocessor
from config import Config

class FeatureExtractor:
    def __init__(self):
        self.preprocessor = AudioPreprocessor()
    
    def extract_optimized_features(self, audio_file, speaker_profile):
        try:
            y, sr = librosa.load(audio_file, sr=22050, duration=4.0)
            
            if len(y) == 0:
                return None
            
            y = self.preprocessor.enhanced_audio_preprocessing(y, sr, speaker_profile)
            features = {}
            
            self._extract_mfcc_features(y, sr, features)
            self._extract_spectral_features(y, sr, features)
            self._extract_f0_features(y, sr, speaker_profile, features)
            self._extract_temporal_features(y, sr, features)
            self._extract_harmonic_features(y, sr, features)
            self._extract_down_syndrome_features(y, sr, speaker_profile, features)
            
            return self._clean_features(features)
            
        except Exception as e:
            print(f"Error extracting features from {audio_file}: {e}")
            return None
    
    def _extract_mfcc_features(self, y, sr, features):
        try:
            mfcc_configs = [
                {'n_mfcc': 13, 'n_fft': 2048, 'hop_length': 512, 'n_mels': 40},
                {'n_mfcc': 20, 'n_fft': 1024, 'hop_length': 256, 'n_mels': 64},
                {'n_mfcc': 26, 'n_fft': 4096, 'hop_length': 1024, 'n_mels': 80}
            ]
            
            for i, config in enumerate(mfcc_configs):
                try:
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, **config)
                    
                    for j in range(min(8, mfcc.shape[0])):
                        coeff_data = mfcc[j]
                        if len(coeff_data) > 0:
                            prefix = f'mfcc_{i}_{j}'
                            features[f'{prefix}_mean'] = float(np.mean(coeff_data))
                            features[f'{prefix}_std'] = float(np.std(coeff_data))
                            features[f'{prefix}_max'] = float(np.max(coeff_data))
                            features[f'{prefix}_min'] = float(np.min(coeff_data))
                            features[f'{prefix}_range'] = features[f'{prefix}_max'] - features[f'{prefix}_min']
                    
                    if mfcc.shape[1] > 10:
                        try:
                            width = min(7, max(3, mfcc.shape[1]//3))
                            if width % 2 == 0:
                                width += 1
                            mfcc_delta = librosa.feature.delta(mfcc, width=width)
                            
                            for j in range(min(5, mfcc_delta.shape[0])):
                                delta_data = mfcc_delta[j]
                                if len(delta_data) > 0:
                                    prefix = f'mfcc_delta_{i}_{j}'
                                    features[f'{prefix}_mean'] = float(np.mean(delta_data))
                                    features[f'{prefix}_std'] = float(np.std(delta_data))
                        except:
                            pass
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
    
    def _extract_spectral_features(self, y, sr, features):
        try:
            spectral_features = []
            
            try:
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_features.append(('centroid', centroid))
            except:
                pass
            
            try:
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
                spectral_features.append(('rolloff', rolloff))
            except:
                pass
            
            try:
                bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                spectral_features.append(('bandwidth', bandwidth))
            except:
                pass
            
            try:
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                spectral_features.append(('zcr', zcr))
            except:
                pass
            
            try:
                rms = librosa.feature.rms(y=y)[0]
                spectral_features.append(('rms', rms))
            except:
                pass
            
            for name, values in spectral_features:
                if len(values) > 0:
                    features[f'{name}_mean'] = float(np.mean(values))
                    features[f'{name}_std'] = float(np.std(values))
                    features[f'{name}_max'] = float(np.max(values))
                    features[f'{name}_min'] = float(np.min(values))
                    features[f'{name}_median'] = float(np.median(values))
                    features[f'{name}_q75'] = float(np.percentile(values, 75))
                    features[f'{name}_q25'] = float(np.percentile(values, 25))
                    features[f'{name}_iqr'] = features[f'{name}_q75'] - features[f'{name}_q25']
                    
                    if len(values) > 1:
                        features[f'{name}_skew'] = float(skew(values))
                        features[f'{name}_slope'] = float(np.polyfit(range(len(values)), values, 1)[0])
                    else:
                        features[f'{name}_skew'] = 0.0
                        features[f'{name}_slope'] = 0.0
            
            try:
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, n_fft=2048)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                for i, (start, end) in enumerate([(0, 8), (8, 16), (16, 24), (24, 32)]):
                    band_data = mel_spec_db[start:end].mean(axis=0)
                    if len(band_data) > 0:
                        features[f'mel_band_{i}_mean'] = float(np.mean(band_data))
                        features[f'mel_band_{i}_std'] = float(np.std(band_data))
                        features[f'mel_band_{i}_energy'] = float(np.sum(band_data**2))
            except:
                pass
                
        except Exception as e:
            pass
    
    def _extract_f0_features(self, y, sr, speaker_profile, features):
        try:
            clarity = speaker_profile.get("clarity", 0.5)
            
            if clarity > 0.7:
                f0_configs = [
                    {'fmin': 80, 'fmax': 350},
                    {'fmin': 100, 'fmax': 300}
                ]
            else:
                f0_configs = [
                    {'fmin': 70, 'fmax': 400},
                    {'fmin': 90, 'fmax': 350}
                ]
            
            for i, config in enumerate(f0_configs):
                try:
                    f0 = librosa.yin(y, sr=sr, fmin=config['fmin'], fmax=config['fmax'])
                    f0_clean = f0[f0 > 0]
                    
                    if len(f0_clean) > 0:
                        prefix = f'f0_{i}'
                        
                        features[f'{prefix}_mean'] = float(np.mean(f0_clean))
                        features[f'{prefix}_std'] = float(np.std(f0_clean))
                        features[f'{prefix}_max'] = float(np.max(f0_clean))
                        features[f'{prefix}_min'] = float(np.min(f0_clean))
                        features[f'{prefix}_range'] = features[f'{prefix}_max'] - features[f'{prefix}_min']
                        features[f'{prefix}_median'] = float(np.median(f0_clean))
                        features[f'{prefix}_voiced_ratio'] = len(f0_clean) / len(f0)
                        
                        if len(f0_clean) > 1:
                            f0_diff = np.diff(f0_clean)
                            features[f'{prefix}_jitter'] = float(np.std(f0_diff) / np.mean(f0_clean)) if np.mean(f0_clean) > 0 else 0
                            features[f'{prefix}_smoothness'] = float(1 / (1 + np.std(f0_diff)))
                            features[f'{prefix}_slope'] = float(np.polyfit(range(len(f0_clean)), f0_clean, 1)[0])
                            
                            rolling_std = np.convolve(f0_clean, np.ones(min(5, len(f0_clean)))/min(5, len(f0_clean)), mode='valid')
                            features[f'{prefix}_stability'] = float(1 / (1 + np.std(rolling_std)))
                        else:
                            for feat in ['jitter', 'smoothness', 'slope', 'stability']:
                                features[f'{prefix}_{feat}'] = 0.0
                    else:
                        for feat in ['mean', 'std', 'max', 'min', 'range', 'median', 'voiced_ratio', 'jitter', 'smoothness', 'slope', 'stability']:
                            features[f'f0_{i}_{feat}'] = 0.0
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
    
    def _extract_temporal_features(self, y, sr, features):
        try:
            duration = len(y) / sr
            features['duration'] = duration
            features['sample_rate'] = sr
            features['total_samples'] = len(y)
            features['speech_rate'] = len(y) / sr / duration if duration > 0 else 0
            
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = float(tempo)
                features['beat_count'] = len(beats)
                features['beat_density'] = len(beats) / duration if duration > 0 else 0
                
                if len(beats) > 1:
                    beat_times = librosa.frames_to_time(beats, sr=sr)
                    beat_intervals = np.diff(beat_times)
                    features['beat_regularity'] = float(1 / (1 + np.std(beat_intervals))) if len(beat_intervals) > 0 else 0
                    features['rhythm_strength'] = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
                else:
                    features['beat_regularity'] = 0.0
                    features['rhythm_strength'] = 0.0
            except:
                for feat in ['tempo', 'beat_count', 'beat_density', 'beat_regularity', 'rhythm_strength']:
                    features[feat] = 0.0
            
            try:
                rms_values = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
                if len(rms_values) > 0:
                    silence_thresholds = [np.percentile(rms_values, 10), np.percentile(rms_values, 20)]
                    
                    for i, threshold in enumerate(silence_thresholds):
                        silence_frames = rms_values < threshold
                        features[f'silence_ratio_{i}'] = np.sum(silence_frames) / len(silence_frames)
                        
                        silence_changes = np.diff(silence_frames.astype(int))
                        features[f'silence_segments_{i}'] = np.sum(silence_changes == 1)
                else:
                    for i in range(2):
                        features[f'silence_ratio_{i}'] = 0.0
                        features[f'silence_segments_{i}'] = 0.0
            except:
                for i in range(2):
                    features[f'silence_ratio_{i}'] = 0.0
                    features[f'silence_segments_{i}'] = 0.0
                    
        except Exception as e:
            pass
    
    def _extract_harmonic_features(self, y, sr, features):
        try:
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048)
                if chroma.shape[1] > 0:
                    chroma_mean = np.mean(chroma, axis=1)
                    features['chroma_energy'] = float(np.sum(chroma_mean**2))
                    features['chroma_centroid'] = float(np.argmax(chroma_mean))
                    features['chroma_spread'] = float(np.std(chroma_mean))
                    
                    top_notes = np.argsort(chroma_mean)[-3:]
                    for i, note in enumerate(top_notes):
                        features[f'top_note_{i}'] = float(note)
                        features[f'top_note_{i}_strength'] = float(chroma_mean[note])
            except:
                for feat in ['chroma_energy', 'chroma_centroid', 'chroma_spread']:
                    features[feat] = 0.0
                for i in range(3):
                    features[f'top_note_{i}'] = 0.0
                    features[f'top_note_{i}_strength'] = 0.0
            
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(y, margin=2.0)
                
                harmonic_energy = np.sum(y_harmonic**2)
                percussive_energy = np.sum(y_percussive**2)
                total_energy = harmonic_energy + percussive_energy
                
                if total_energy > 0:
                    features['harmonic_ratio'] = harmonic_energy / total_energy
                    features['percussive_ratio'] = percussive_energy / total_energy
                    features['harmonicity'] = harmonic_energy / (percussive_energy + 1e-10)
                    
                    harmonic_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
                    features['harmonic_centroid_mean'] = float(np.mean(harmonic_centroid)) if len(harmonic_centroid) > 0 else 0
                    
                    percussive_zcr = librosa.feature.zero_crossing_rate(y_percussive)[0]
                    features['percussive_zcr_mean'] = float(np.mean(percussive_zcr)) if len(percussive_zcr) > 0 else 0
                else:
                    for feat in ['harmonic_ratio', 'percussive_ratio', 'harmonicity', 'harmonic_centroid_mean', 'percussive_zcr_mean']:
                        features[feat] = 0.5 if 'ratio' in feat else 0.0
            except:
                for feat in ['harmonic_ratio', 'percussive_ratio', 'harmonicity', 'harmonic_centroid_mean', 'percussive_zcr_mean']:
                    features[feat] = 0.5 if 'ratio' in feat else 0.0
                    
        except Exception as e:
            pass
    
    def _extract_down_syndrome_features(self, y, sr, speaker_profile, features):
        try:
            clarity = speaker_profile.get("clarity", 0.5)
            quality = speaker_profile.get("quality", "متوسط")
            
            features['speaker_clarity'] = clarity
            features['speaker_quality_score'] = Config.QUALITY_MAPPING.get(quality, 0.5)
            
            try:
                spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
                if len(spectral_flatness) > 0:
                    features['speech_clarity_est'] = 1 - np.mean(spectral_flatness)
                    features['speech_consistency'] = 1 / (1 + np.std(spectral_flatness))
                else:
                    features['speech_clarity_est'] = clarity
                    features['speech_consistency'] = 0.5
            except:
                features['speech_clarity_est'] = clarity
                features['speech_consistency'] = 0.5
            
            voice_quality_indicators = [
                features.get('f0_0_jitter', 0.5),
                features.get('f0_0_stability', 0.5),
                features.get('silence_ratio_0', 0.5),
                1 - features.get('speech_clarity_est', 0.5)
            ]
            features['speech_difficulty'] = np.mean(voice_quality_indicators)
            
            performance_indicators = [
                features['speaker_clarity'],
                features['speech_clarity_est'],
                1 - features['speech_difficulty'],
                features.get('harmonic_ratio', 0.5),
                features.get('f0_0_voiced_ratio', 0.5)
            ]
            features['expected_performance'] = np.mean(performance_indicators)
            
        except Exception as e:
            features['speech_difficulty'] = 0.5
            features['expected_performance'] = 0.5
    
    def _clean_features(self, features):
        clean_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float, np.number)):
                if np.isnan(value) or np.isinf(value):
                    clean_features[key] = 0.0
                else:
                    clean_features[key] = float(value)
            else:
                clean_features[key] = 0.0
        
        return clean_features