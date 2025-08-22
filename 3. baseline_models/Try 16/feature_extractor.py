import numpy as np
import librosa
from scipy.stats import skew, kurtosis
from config import FEATURE_CONFIGS

class FeatureExtractor:
    def __init__(self):
        self.configs = FEATURE_CONFIGS
    
    def extract_features(self, audio_file, max_duration=3.0):
        try:
            y, sr = librosa.load(audio_file, sr=22050, duration=max_duration)
            
            if len(y) == 0:
                return None
            
            y = librosa.util.normalize(y)
            features = {}
            
            self._extract_mfcc_features(y, sr, features)
            self._extract_spectral_features(y, sr, features)
            self._extract_prosodic_features(y, sr, features)
            self._extract_temporal_features(y, sr, features)
            self._extract_harmonic_features(y, sr, features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_file}: {e}")
            return None
    
    def _extract_mfcc_features(self, y, sr, features):
        try:
            for i, config in enumerate(self.configs['mfcc_configs']):
                try:
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, **config)
                    mfcc_delta = librosa.feature.delta(mfcc)
                    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                    
                    for j in range(min(13, mfcc.shape[0])):
                        prefix = f'mfcc{config["n_mfcc"]}_{j}'
                        features[f'{prefix}_mean'] = np.mean(mfcc[j])
                        features[f'{prefix}_std'] = np.std(mfcc[j])
                        features[f'{prefix}_max'] = np.max(mfcc[j])
                        features[f'{prefix}_min'] = np.min(mfcc[j])
                        features[f'{prefix}_median'] = np.median(mfcc[j])
                        features[f'{prefix}_q25'] = np.percentile(mfcc[j], 25)
                        features[f'{prefix}_q75'] = np.percentile(mfcc[j], 75)
                        features[f'{prefix}_iqr'] = np.percentile(mfcc[j], 75) - np.percentile(mfcc[j], 25)
                        features[f'{prefix}_skew'] = skew(mfcc[j])
                        features[f'{prefix}_kurtosis'] = kurtosis(mfcc[j])
                        features[f'{prefix}_delta_mean'] = np.mean(mfcc_delta[j])
                        features[f'{prefix}_delta_std'] = np.std(mfcc_delta[j])
                        features[f'{prefix}_delta2_mean'] = np.mean(mfcc_delta2[j])
                        features[f'{prefix}_delta2_std'] = np.std(mfcc_delta2[j])
                except:
                    continue
        except Exception as e:
            print(f"MFCC error: {e}")
            for i in range(200):
                features[f'mfcc_feat_{i}'] = 0
    
    def _extract_spectral_features(self, y, sr, features):
        try:
            for i, config in enumerate(self.configs['mel_configs']):
                try:
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, **config)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    for j in range(min(20, mel_spec_db.shape[0])):
                        prefix = f'mel{config["n_mels"]}_{j}'
                        features[f'{prefix}_mean'] = np.mean(mel_spec_db[j])
                        features[f'{prefix}_std'] = np.std(mel_spec_db[j])
                        features[f'{prefix}_max'] = np.max(mel_spec_db[j])
                        features[f'{prefix}_min'] = np.min(mel_spec_db[j])
                        features[f'{prefix}_range'] = np.max(mel_spec_db[j]) - np.min(mel_spec_db[j])
                except:
                    continue
            
            spectral_features = [
                ('spectral_centroid', librosa.feature.spectral_centroid(y=y, sr=sr)[0]),
                ('spectral_rolloff', librosa.feature.spectral_rolloff(y=y, sr=sr)[0]),
                ('spectral_bandwidth', librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]),
                ('spectral_flatness', librosa.feature.spectral_flatness(y=y)[0]),
                ('spectral_contrast', np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=0)),
                ('zcr', librosa.feature.zero_crossing_rate(y)[0]),
                ('rms', librosa.feature.rms(y=y)[0])
            ]
            
            for name, values in spectral_features:
                if len(values.shape) == 0:
                    values = [values]
                features[f'{name}_mean'] = np.mean(values)
                features[f'{name}_std'] = np.std(values)
                features[f'{name}_max'] = np.max(values)
                features[f'{name}_min'] = np.min(values)
                features[f'{name}_median'] = np.median(values)
                features[f'{name}_skew'] = skew(values)
                features[f'{name}_kurtosis'] = kurtosis(values)
                features[f'{name}_range'] = np.max(values) - np.min(values)
                
        except Exception as e:
            print(f"Spectral error: {e}")
            for i in range(150):
                features[f'spectral_feat_{i}'] = 0
    
    def _extract_prosodic_features(self, y, sr, features):
        try:
            for i, params in enumerate(self.configs['f0_methods']):
                try:
                    f0 = librosa.yin(y, sr=sr, **params)
                    f0_clean = f0[f0 > 0]
                    
                    if len(f0_clean) > 0:
                        prefix = f'f0_{i}'
                        features[f'{prefix}_mean'] = np.mean(f0_clean)
                        features[f'{prefix}_std'] = np.std(f0_clean)
                        features[f'{prefix}_max'] = np.max(f0_clean)
                        features[f'{prefix}_min'] = np.min(f0_clean)
                        features[f'{prefix}_range'] = np.max(f0_clean) - np.min(f0_clean)
                        features[f'{prefix}_median'] = np.median(f0_clean)
                        features[f'{prefix}_q25'] = np.percentile(f0_clean, 25)
                        features[f'{prefix}_q75'] = np.percentile(f0_clean, 75)
                        features[f'{prefix}_iqr'] = np.percentile(f0_clean, 75) - np.percentile(f0_clean, 25)
                        features[f'{prefix}_voiced_ratio'] = len(f0_clean) / len(f0)
                        features[f'{prefix}_skew'] = skew(f0_clean)
                        features[f'{prefix}_kurtosis'] = kurtosis(f0_clean)
                        
                        if len(f0_clean) > 1:
                            f0_diff = np.diff(f0_clean)
                            features[f'{prefix}_jitter'] = np.mean(np.abs(f0_diff)) / np.mean(f0_clean)
                            features[f'{prefix}_slope'] = np.polyfit(range(len(f0_clean)), f0_clean, 1)[0]
                    else:
                        for feat in ['mean', 'std', 'max', 'min', 'range', 'median', 'q25', 'q75', 'iqr', 'voiced_ratio', 'skew', 'kurtosis', 'jitter', 'slope']:
                            features[f'f0_{i}_{feat}'] = 0
                except:
                    continue
            
            window_sizes = [512, 1024, 2048]
            for i, win_size in enumerate(window_sizes):
                try:
                    rms_energy = librosa.feature.rms(y=y, frame_length=win_size, hop_length=win_size//4)[0]
                    prefix = f'energy_{i}'
                    features[f'{prefix}_mean'] = np.mean(rms_energy)
                    features[f'{prefix}_std'] = np.std(rms_energy)
                    features[f'{prefix}_max'] = np.max(rms_energy)
                    features[f'{prefix}_min'] = np.min(rms_energy)
                    features[f'{prefix}_range'] = np.max(rms_energy) - np.min(rms_energy)
                    features[f'{prefix}_skew'] = skew(rms_energy)
                    features[f'{prefix}_kurtosis'] = kurtosis(rms_energy)
                    
                    if len(rms_energy) > 1:
                        energy_diff = np.abs(np.diff(rms_energy))
                        features[f'{prefix}_shimmer'] = np.mean(energy_diff) / np.mean(rms_energy) if np.mean(rms_energy) > 0 else 0
                except:
                    continue
                    
        except Exception as e:
            print(f"Prosodic error: {e}")
            for i in range(100):
                features[f'prosodic_feat_{i}'] = 0
    
    def _extract_temporal_features(self, y, sr, features):
        try:
            duration = len(y) / sr
            features['duration'] = duration
            features['sample_rate'] = sr
            features['total_samples'] = len(y)
            
            for i, params in enumerate(self.configs['onset_methods']):
                try:
                    if params['units'] == 'frames':
                        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, **{k:v for k,v in params.items() if k != 'units'})
                        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                    else:
                        onset_times = librosa.onset.onset_detect(y=y, sr=sr, **params)
                    
                    prefix = f'onset_{i}'
                    features[f'{prefix}_count'] = len(onset_times)
                    features[f'{prefix}_rate'] = len(onset_times) / duration if duration > 0 else 0
                    
                    if len(onset_times) > 1:
                        onset_intervals = np.diff(onset_times)
                        features[f'{prefix}_interval_mean'] = np.mean(onset_intervals)
                        features[f'{prefix}_interval_std'] = np.std(onset_intervals)
                        features[f'{prefix}_interval_median'] = np.median(onset_intervals)
                        features[f'{prefix}_regularity'] = np.std(onset_intervals) / np.mean(onset_intervals) if np.mean(onset_intervals) > 0 else 0
                    else:
                        for feat in ['interval_mean', 'interval_std', 'interval_median', 'regularity']:
                            features[f'{prefix}_{feat}'] = 0
                except:
                    continue
            
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = tempo
                features['beat_count'] = len(beats)
                
                if len(beats) > 1:
                    beat_intervals = np.diff(beats)
                    features['beat_regularity'] = np.std(beat_intervals) / np.mean(beat_intervals) if np.mean(beat_intervals) > 0 else 0
                    features['rhythm_strength'] = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
                else:
                    features['beat_regularity'] = 0
                    features['rhythm_strength'] = 0
            except:
                features['tempo'] = 0
                features['beat_count'] = 0
                features['beat_regularity'] = 0
                features['rhythm_strength'] = 0
            
            try:
                rms_values = librosa.feature.rms(y=y)[0]
                silence_thresholds = [0.01, 0.02, 0.05]
                
                for i, threshold in enumerate(silence_thresholds):
                    silence_frames = rms_values < threshold
                    features[f'silence_ratio_{i}'] = np.sum(silence_frames) / len(silence_frames)
                    
                    silence_segments = []
                    in_silence = False
                    segment_start = 0
                    
                    for j, is_silent in enumerate(silence_frames):
                        if is_silent and not in_silence:
                            in_silence = True
                            segment_start = j
                        elif not is_silent and in_silence:
                            in_silence = False
                            silence_segments.append(j - segment_start)
                    
                    if silence_segments:
                        features[f'silence_segments_{i}'] = len(silence_segments)
                        features[f'avg_silence_length_{i}'] = np.mean(silence_segments)
                    else:
                        features[f'silence_segments_{i}'] = 0
                        features[f'avg_silence_length_{i}'] = 0
            except:
                for i in range(3):
                    features[f'silence_ratio_{i}'] = 0
                    features[f'silence_segments_{i}'] = 0
                    features[f'avg_silence_length_{i}'] = 0
                    
        except Exception as e:
            print(f"Temporal error: {e}")
            for i in range(80):
                features[f'temporal_feat_{i}'] = 0
    
    def _extract_harmonic_features(self, y, sr, features):
        try:
            for i, config in enumerate(self.configs['chroma_configs']):
                try:
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr, **config)
                    for j in range(12):
                        prefix = f'chroma_{i}_{j}'
                        features[f'{prefix}_mean'] = np.mean(chroma[j])
                        features[f'{prefix}_std'] = np.std(chroma[j])
                        features[f'{prefix}_max'] = np.max(chroma[j])
                except:
                    continue
            
            margins = [1.0, 2.0, 4.0]
            for i, margin in enumerate(margins):
                try:
                    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=margin)
                    
                    harmonic_energy = np.sum(y_harmonic**2)
                    percussive_energy = np.sum(y_percussive**2)
                    total_energy = harmonic_energy + percussive_energy
                    
                    if total_energy > 0:
                        prefix = f'hpss_{i}'
                        features[f'{prefix}_harmonic_ratio'] = harmonic_energy / total_energy
                        features[f'{prefix}_percussive_ratio'] = percussive_energy / total_energy
                        features[f'{prefix}_hp_ratio'] = harmonic_energy / percussive_energy if percussive_energy > 0 else 0
                    else:
                        for feat in ['harmonic_ratio', 'percussive_ratio', 'hp_ratio']:
                            features[f'hpss_{i}_{feat}'] = 0
                except:
                    continue
            
            try:
                y_harmonic, _ = librosa.effects.hpss(y)
                tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
                for i in range(6):
                    features[f'tonnetz_{i}_mean'] = np.mean(tonnetz[i])
                    features[f'tonnetz_{i}_std'] = np.std(tonnetz[i])
                    features[f'tonnetz_{i}_max'] = np.max(tonnetz[i])
                    features[f'tonnetz_{i}_min'] = np.min(tonnetz[i])
            except:
                for i in range(6):
                    for stat in ['mean', 'std', 'max', 'min']:
                        features[f'tonnetz_{i}_{stat}'] = 0
                        
        except Exception as e:
            print(f"Harmonic error: {e}")
            for i in range(70):
                features[f'harmonic_feat_{i}'] = 0
