#!/usr/bin/env python3

import numpy as np
import librosa
from scipy.stats import skew, kurtosis
from scipy import signal
from collections import Counter
from pathlib import Path
from utils import UtilsHelper

class FeatureExtractor:
    
    def __init__(self):
        self.utils = UtilsHelper()
    
    def extract_comprehensive_features(self, audio_file, speaker_profile, word_categories, difficulty_levels, word_quality_map):
        try:
            y, sr = librosa.load(audio_file, sr=22050, duration=5.0)
            
            if len(y) == 0:
                return None
            
            word = audio_file.parent.name
            word_info = {
                'difficulty': self.utils.get_word_difficulty_advanced(word, difficulty_levels),
                'category': self.utils.get_word_category(word, word_categories),
                'quality': self.utils.get_word_quality_for_speaker(word, speaker_profile.get('name', 'Unknown'), word_quality_map)
            }
            
            from audio_processor import AudioProcessor
            processor = AudioProcessor()
            y = processor.enhanced_audio_preprocessing_v2(y, sr, speaker_profile, word_info)
            
            features = {}
            
            features.update(self.extract_advanced_mfcc_features(y, sr, speaker_profile))
            features.update(self.extract_comprehensive_spectral_features(y, sr))
            features.update(self.extract_advanced_f0_features(y, sr, speaker_profile))
            features.update(self.extract_comprehensive_temporal_features(y, sr))
            features.update(self.extract_advanced_harmonic_features(y, sr))
            features.update(self.extract_down_syndrome_specific_features(y, sr, speaker_profile, word_info))
            features.update(self.extract_statistical_features(y, sr))
            features.update(self.extract_complexity_features(y, sr))
            
            clean_features = self.clean_features(features)
            
            return clean_features
            
        except Exception as e:
            print(f"Feature extraction error for {audio_file}: {e}")
            return None
    
    def extract_advanced_mfcc_features(self, y, sr, speaker_profile):
        features = {}
        try:
            mfcc_configs = [
                {'n_mfcc': 13, 'n_fft': 2048, 'hop_length': 512, 'n_mels': 40},
                {'n_mfcc': 20, 'n_fft': 1024, 'hop_length': 256, 'n_mels': 64},
                {'n_mfcc': 26, 'n_fft': 4096, 'hop_length': 1024, 'n_mels': 80},
                {'n_mfcc': 15, 'n_fft': 1536, 'hop_length': 384, 'n_mels': 50}
            ]
            
            for i, config in enumerate(mfcc_configs):
                try:
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, **config)
                    
                    for j in range(min(12, mfcc.shape[0])):
                        coeff_data = mfcc[j]
                        if len(coeff_data) > 0:
                            prefix = f'mfcc_{i}_{j}'
                            
                            features[f'{prefix}_mean'] = float(np.mean(coeff_data))
                            features[f'{prefix}_std'] = float(np.std(coeff_data))
                            features[f'{prefix}_max'] = float(np.max(coeff_data))
                            features[f'{prefix}_min'] = float(np.min(coeff_data))
                            features[f'{prefix}_range'] = features[f'{prefix}_max'] - features[f'{prefix}_min']
                            features[f'{prefix}_median'] = float(np.median(coeff_data))
                            
                            if len(coeff_data) > 1:
                                features[f'{prefix}_skew'] = float(skew(coeff_data))
                                features[f'{prefix}_kurtosis'] = float(kurtosis(coeff_data))
                                features[f'{prefix}_iqr'] = float(np.percentile(coeff_data, 75) - np.percentile(coeff_data, 25))
                                features[f'{prefix}_energy'] = float(np.sum(coeff_data**2))
                    
                    if mfcc.shape[1] > 15:
                        try:
                            for delta_order, delta_name in [(1, 'delta'), (2, 'delta2')]:
                                width = min(9, max(3, mfcc.shape[1]//4))
                                if width % 2 == 0:
                                    width += 1
                                
                                mfcc_delta = librosa.feature.delta(mfcc, width=width, order=delta_order)
                                
                                for j in range(min(8, mfcc_delta.shape[0])):
                                    delta_data = mfcc_delta[j]
                                    if len(delta_data) > 0:
                                        prefix = f'mfcc_{delta_name}_{i}_{j}'
                                        features[f'{prefix}_mean'] = float(np.mean(delta_data))
                                        features[f'{prefix}_std'] = float(np.std(delta_data))
                                        features[f'{prefix}_energy'] = float(np.sum(delta_data**2))
                        except:
                            pass
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
        
        return features
    
    def extract_comprehensive_spectral_features(self, y, sr):
        features = {}
        try:
            spectral_features = []
            
            try:
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_features.append(('centroid', centroid))
            except: pass
            
            try:
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
                spectral_features.append(('rolloff', rolloff))
            except: pass
            
            try:
                rolloff95 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)[0]
                spectral_features.append(('rolloff95', rolloff95))
            except: pass
            
            try:
                bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                spectral_features.append(('bandwidth', bandwidth))
            except: pass
            
            try:
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                spectral_features.append(('zcr', zcr))
            except: pass
            
            try:
                rms = librosa.feature.rms(y=y)[0]
                spectral_features.append(('rms', rms))
            except: pass
            
            try:
                flatness = librosa.feature.spectral_flatness(y=y)[0]
                spectral_features.append(('flatness', flatness))
            except: pass
            
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
                    features[f'{name}_range'] = features[f'{name}_max'] - features[f'{name}_min']
                    
                    if len(values) > 1:
                        features[f'{name}_skew'] = float(skew(values))
                        features[f'{name}_kurtosis'] = float(kurtosis(values))
                        features[f'{name}_slope'] = float(np.polyfit(range(len(values)), values, 1)[0])
                        features[f'{name}_energy'] = float(np.sum(values**2))
                        
                        diff_values = np.diff(values)
                        features[f'{name}_trend_strength'] = float(np.std(diff_values))
                        features[f'{name}_stability'] = float(1 / (1 + np.std(diff_values)))
            
            try:
                mel_configs = [
                    {'n_mels': 32, 'n_fft': 2048},
                    {'n_mels': 64, 'n_fft': 1024},
                    {'n_mels': 128, 'n_fft': 4096}
                ]
                
                for i, config in enumerate(mel_configs):
                    try:
                        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, **config)
                        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                        
                        n_bands = 8
                        band_size = mel_spec_db.shape[0] // n_bands
                        
                        for band in range(n_bands):
                            start_idx = band * band_size
                            end_idx = min((band + 1) * band_size, mel_spec_db.shape[0])
                            
                            if start_idx < end_idx:
                                band_data = mel_spec_db[start_idx:end_idx].mean(axis=0)
                                if len(band_data) > 0:
                                    features[f'mel_{i}_band_{band}_mean'] = float(np.mean(band_data))
                                    features[f'mel_{i}_band_{band}_std'] = float(np.std(band_data))
                                    features[f'mel_{i}_band_{band}_energy'] = float(np.sum(band_data**2))
                                    features[f'mel_{i}_band_{band}_max'] = float(np.max(band_data))
                    except:
                        continue
            except:
                pass
                
        except Exception as e:
            pass
        
        return features
    
    def extract_advanced_f0_features(self, y, sr, speaker_profile):
        features = {}
        try:
            clarity = speaker_profile.get("clarity", 0.5)
            age = speaker_profile.get("age", "5").split("-")[0]
            age_num = int(age) if age.isdigit() else 5
            
            if age_num < 10:
                base_configs = [
                    {'fmin': 100, 'fmax': 400},
                    {'fmin': 120, 'fmax': 350},
                    {'fmin': 80, 'fmax': 450}
                ]
            elif age_num > 20:
                base_configs = [
                    {'fmin': 70, 'fmax': 300},
                    {'fmin': 60, 'fmax': 250},
                    {'fmin': 80, 'fmax': 350}
                ]
            else:
                base_configs = [
                    {'fmin': 80, 'fmax': 350},
                    {'fmin': 90, 'fmax': 300},
                    {'fmin': 70, 'fmax': 400}
                ]
            
            f0_configs = []
            for config in base_configs:
                if clarity < 0.6:
                    config['fmin'] = max(50, config['fmin'] - 20)
                    config['fmax'] = config['fmax'] + 50
                f0_configs.append(config)
            
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
                            features[f'{prefix}_skew'] = float(skew(f0_clean))
                            features[f'{prefix}_kurtosis'] = float(kurtosis(f0_clean))
                            
                            f0_diff = np.diff(f0_clean)
                            features[f'{prefix}_jitter'] = float(np.std(f0_diff) / np.mean(f0_clean)) if np.mean(f0_clean) > 0 else 0
                            features[f'{prefix}_smoothness'] = float(1 / (1 + np.std(f0_diff)))
                            features[f'{prefix}_variability'] = float(np.std(f0_clean) / np.mean(f0_clean)) if np.mean(f0_clean) > 0 else 0
                            
                            features[f'{prefix}_slope'] = float(np.polyfit(range(len(f0_clean)), f0_clean, 1)[0])
                            
                            if len(f0_clean) >= 5:
                                window_size = min(5, len(f0_clean))
                                rolling_std = []
                                for j in range(len(f0_clean) - window_size + 1):
                                    window = f0_clean[j:j + window_size]
                                    rolling_std.append(np.std(window))
                                
                                if rolling_std:
                                    features[f'{prefix}_stability'] = float(1 / (1 + np.mean(rolling_std)))
                                    features[f'{prefix}_stability_var'] = float(np.var(rolling_std))
                            
                            f0_normalized = (f0_clean - np.mean(f0_clean)) / (np.std(f0_clean) + 1e-10)
                            if len(f0_normalized) > 10:
                                try:
                                    fft_f0 = np.abs(np.fft.fft(f0_normalized))[:len(f0_normalized)//2]
                                    if len(fft_f0) > 0:
                                        features[f'{prefix}_spectral_energy'] = float(np.sum(fft_f0**2))
                                        features[f'{prefix}_spectral_centroid'] = float(np.sum(np.arange(len(fft_f0)) * fft_f0) / (np.sum(fft_f0) + 1e-10))
                                except:
                                    features[f'{prefix}_spectral_energy'] = 0.0
                                    features[f'{prefix}_spectral_centroid'] = 0.0
                        
                        features[f'{prefix}_q25'] = float(np.percentile(f0_clean, 25))
                        features[f'{prefix}_q75'] = float(np.percentile(f0_clean, 75))
                        features[f'{prefix}_iqr'] = features[f'{prefix}_q75'] - features[f'{prefix}_q25']
                        
                    else:
                        default_features = [
                            'mean', 'std', 'max', 'min', 'range', 'median', 'voiced_ratio',
                            'skew', 'kurtosis', 'jitter', 'smoothness', 'variability', 'slope',
                            'stability', 'stability_var', 'spectral_energy', 'spectral_centroid',
                            'q25', 'q75', 'iqr'
                        ]
                        for feat in default_features:
                            features[f'f0_{i}_{feat}'] = 0.0
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
        
        return features
    
    def extract_comprehensive_temporal_features(self, y, sr):
        features = {}
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
                    features['beat_variance'] = float(np.var(beat_intervals)) if len(beat_intervals) > 0 else 0
                    
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                features['onset_count'] = len(onset_times)
                features['onset_rate'] = len(onset_times) / duration if duration > 0 else 0
                
                if len(onset_times) > 1:
                    onset_intervals = np.diff(onset_times)
                    features['onset_regularity'] = float(1 / (1 + np.std(onset_intervals))) if len(onset_intervals) > 0 else 0
                    features['onset_variance'] = float(np.var(onset_intervals)) if len(onset_intervals) > 0 else 0
                
                onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
                if len(onset_strength) > 0:
                    features['rhythm_strength'] = float(np.mean(onset_strength))
                    features['rhythm_variance'] = float(np.var(onset_strength))
                    features['rhythm_max'] = float(np.max(onset_strength))
                
            except:
                for feat in ['tempo', 'beat_count', 'beat_density', 'beat_regularity', 'beat_variance',
                           'onset_count', 'onset_rate', 'onset_regularity', 'onset_variance',
                           'rhythm_strength', 'rhythm_variance', 'rhythm_max']:
                    features[feat] = 0.0
            
            try:
                frame_lengths = [512, 1024, 2048]
                hop_length = 256
                
                for i, frame_length in enumerate(frame_lengths):
                    try:
                        rms_values = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                        if len(rms_values) > 0:
                            silence_thresholds = [
                                np.percentile(rms_values, 10),
                                np.percentile(rms_values, 20),
                                np.percentile(rms_values, 30)
                            ]
                            
                            for j, threshold in enumerate(silence_thresholds):
                                silence_frames = rms_values < threshold
                                features[f'silence_ratio_{i}_{j}'] = np.sum(silence_frames) / len(silence_frames)
                                
                                silence_changes = np.diff(silence_frames.astype(int))
                                features[f'silence_segments_{i}_{j}'] = np.sum(silence_changes == 1)
                                
                                if np.sum(silence_changes == 1) > 0:
                                    silence_lengths = []
                                    in_silence = False
                                    current_length = 0
                                    
                                    for frame in silence_frames:
                                        if frame:
                                            if not in_silence:
                                                in_silence = True
                                                current_length = 1
                                            else:
                                                current_length += 1
                                        else:
                                            if in_silence:
                                                silence_lengths.append(current_length)
                                                in_silence = False
                                    
                                    if silence_lengths:
                                        features[f'avg_silence_length_{i}_{j}'] = float(np.mean(silence_lengths))
                                        features[f'max_silence_length_{i}_{j}'] = float(np.max(silence_lengths))
                                    else:
                                        features[f'avg_silence_length_{i}_{j}'] = 0.0
                                        features[f'max_silence_length_{i}_{j}'] = 0.0
                                else:
                                    features[f'avg_silence_length_{i}_{j}'] = 0.0
                                    features[f'max_silence_length_{i}_{j}'] = 0.0
                    except:
                        continue
            except:
                pass
                
        except Exception as e:
            pass
        
        return features
    
    def extract_advanced_harmonic_features(self, y, sr):
        features = {}
        try:
            try:
                chroma_configs = [
                    {'n_fft': 2048, 'hop_length': 512},
                    {'n_fft': 4096, 'hop_length': 1024},
                    {'n_fft': 1024, 'hop_length': 256}
                ]
                
                for i, config in enumerate(chroma_configs):
                    try:
                        chroma = librosa.feature.chroma_stft(y=y, sr=sr, **config)
                        if chroma.shape[1] > 0:
                            chroma_mean = np.mean(chroma, axis=1)
                            
                            features[f'chroma_{i}_energy'] = float(np.sum(chroma_mean**2))
                            features[f'chroma_{i}_centroid'] = float(np.argmax(chroma_mean))
                            features[f'chroma_{i}_spread'] = float(np.std(chroma_mean))
                            features[f'chroma_{i}_skew'] = float(skew(chroma_mean))
                            features[f'chroma_{i}_kurtosis'] = float(kurtosis(chroma_mean))
                            
                            top_notes = np.argsort(chroma_mean)[-5:]
                            for j, note in enumerate(top_notes):
                                features[f'chroma_{i}_top_note_{j}'] = float(note)
                                features[f'chroma_{i}_top_note_{j}_strength'] = float(chroma_mean[note])
                            
                            chroma_var = np.var(chroma, axis=1)
                            features[f'chroma_{i}_harmony'] = float(1 / (1 + np.mean(chroma_var)))
                            features[f'chroma_{i}_consistency'] = float(1 / (1 + np.std(chroma_var)))
                    except:
                        continue
            except:
                pass
            
            try:
                margins = [1.0, 2.0, 3.0]
                for i, margin in enumerate(margins):
                    try:
                        y_harmonic, y_percussive = librosa.effects.hpss(y, margin=margin)
                        
                        harmonic_energy = np.sum(y_harmonic**2)
                        percussive_energy = np.sum(y_percussive**2)
                        total_energy = harmonic_energy + percussive_energy
                        
                        if total_energy > 0:
                            features[f'harmonic_ratio_{i}'] = harmonic_energy / total_energy
                            features[f'percussive_ratio_{i}'] = percussive_energy / total_energy
                            features[f'harmonicity_{i}'] = harmonic_energy / (percussive_energy + 1e-10)
                            
                            try:
                                harmonic_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
                                features[f'harmonic_centroid_mean_{i}'] = float(np.mean(harmonic_centroid)) if len(harmonic_centroid) > 0 else 0
                                features[f'harmonic_centroid_std_{i}'] = float(np.std(harmonic_centroid)) if len(harmonic_centroid) > 0 else 0
                                
                                percussive_zcr = librosa.feature.zero_crossing_rate(y_percussive)[0]
                                features[f'percussive_zcr_mean_{i}'] = float(np.mean(percussive_zcr)) if len(percussive_zcr) > 0 else 0
                                features[f'percussive_zcr_std_{i}'] = float(np.std(percussive_zcr)) if len(percussive_zcr) > 0 else 0
                            except:
                                features[f'harmonic_centroid_mean_{i}'] = 0.0
                                features[f'harmonic_centroid_std_{i}'] = 0.0
                                features[f'percussive_zcr_mean_{i}'] = 0.0
                                features[f'percussive_zcr_std_{i}'] = 0.0
                        else:
                            for feat in ['harmonic_ratio', 'percussive_ratio', 'harmonicity']:
                                features[f'{feat}_{i}'] = 0.5 if 'ratio' in feat else 0.0
                    except:
                        continue
            except:
                pass
                
        except Exception as e:
            pass
        
        return features
    
    def extract_down_syndrome_specific_features(self, y, sr, speaker_profile, word_info):
        features = {}
        try:
            clarity = speaker_profile.get("clarity", 0.5)
            quality = speaker_profile.get("overall_quality", "متوسط")
            age = speaker_profile.get("age", "5").split("-")[0]
            age_num = int(age) if age.isdigit() else 5
            iq = speaker_profile.get("iq", 50)
            
            features['speaker_clarity'] = clarity
            features['speaker_age'] = age_num
            features['speaker_iq'] = iq
            
            quality_mapping = {"ممتاز": 0.9, "جيد": 0.75, "متوسط": 0.6, "ضعيف": 0.4}
            features['speaker_quality_score'] = quality_mapping.get(quality, 0.5)
            
            features['word_difficulty_score'] = self.utils.get_difficulty_score(word_info['difficulty'])
            features['word_category_score'] = self.utils.get_category_score(word_info['category'])
            features['word_quality_score'] = quality_mapping.get(word_info['quality'], 0.5)
            
            try:
                spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
                if len(spectral_flatness) > 0:
                    features['speech_clarity_est'] = 1 - np.mean(spectral_flatness)
                    features['speech_consistency'] = 1 / (1 + np.std(spectral_flatness))
                    features['speech_naturalness'] = float(1 / (1 + np.var(spectral_flatness)))
                else:
                    features['speech_clarity_est'] = clarity
                    features['speech_consistency'] = 0.5
                    features['speech_naturalness'] = 0.5
            except:
                features['speech_clarity_est'] = clarity
                features['speech_consistency'] = 0.5
                features['speech_naturalness'] = 0.5
            
            voice_quality_indicators = [
                features.get('f0_0_jitter', 0.5),
                features.get('f0_0_stability', 0.5),
                features.get('silence_ratio_0_0', 0.5),
                1 - features.get('speech_clarity_est', 0.5),
                features.get('f0_0_variability', 0.5)
            ]
            features['speech_difficulty'] = np.mean(voice_quality_indicators)
            
            performance_indicators = [
                features['speaker_clarity'],
                features['speech_clarity_est'],
                1 - features['speech_difficulty'],
                features.get('harmonic_ratio_0', 0.5),
                features.get('f0_0_voiced_ratio', 0.5),
                features['word_quality_score'],
                1 - features['word_difficulty_score']
            ]
            features['expected_performance'] = np.mean(performance_indicators)
            
            if age_num < 10:
                features['developmental_stage'] = 0.2
            elif age_num < 15:
                features['developmental_stage'] = 0.4
            elif age_num < 25:
                features['developmental_stage'] = 0.6
            else:
                features['developmental_stage'] = 0.8
            
            features['vocal_adaptation'] = (features['speaker_clarity'] + features['speech_clarity_est']) / 2
            
            features['articulation_challenge'] = (
                features['speech_difficulty'] * 0.4 +
                features['word_difficulty_score'] * 0.3 +
                (1 - features['speaker_quality_score']) * 0.3
            )
            
            features['speech_potential'] = (
                features['speaker_iq'] / 100 * 0.3 +
                features['speaker_clarity'] * 0.4 +
                features['developmental_stage'] * 0.3
            )
            
        except Exception as e:
            features['speech_difficulty'] = 0.5
            features['expected_performance'] = 0.5
            features['vocal_adaptation'] = 0.5
            features['articulation_challenge'] = 0.5
            features['speech_potential'] = 0.5
        
        return features
    
    def extract_statistical_features(self, y, sr):
        features = {}
        try:
            features['signal_mean'] = float(np.mean(y))
            features['signal_std'] = float(np.std(y))
            features['signal_var'] = float(np.var(y))
            features['signal_skew'] = float(skew(y))
            features['signal_kurtosis'] = float(kurtosis(y))
            features['signal_entropy'] = self.calculate_entropy(y)
            
            features['energy_total'] = float(np.sum(y**2))
            features['energy_mean'] = float(np.mean(y**2))
            features['energy_std'] = float(np.std(y**2))
            
            y_abs = np.abs(y)
            features['abs_mean'] = float(np.mean(y_abs))
            features['abs_std'] = float(np.std(y_abs))
            features['abs_max'] = float(np.max(y_abs))
            
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                features[f'percentile_{p}'] = float(np.percentile(y_abs, p))
            
            y_diff = np.diff(y)
            if len(y_diff) > 0:
                features['diff_mean'] = float(np.mean(y_diff))
                features['diff_std'] = float(np.std(y_diff))
                features['diff_var'] = float(np.var(y_diff))
                features['signal_smoothness'] = float(1 / (1 + np.std(y_diff)))
            
            if len(y) > 10:
                window_size = min(10, len(y) // 10)
                window_stds = []
                for i in range(0, len(y) - window_size, window_size):
                    window = y[i:i + window_size]
                    window_stds.append(np.std(window))
                
                if window_stds:
                    features['local_stability'] = float(1 / (1 + np.std(window_stds)))
                    features['global_stability'] = float(1 / (1 + np.mean(window_stds)))
            
        except Exception as e:
            pass
        
        return features
    
    def extract_complexity_features(self, y, sr):
        features = {}
        try:
            features['sample_entropy'] = self.sample_entropy(y)
            
            features['approximate_entropy'] = self.approximate_entropy(y)
            
            try:
                stft = librosa.stft(y)
                magnitude = np.abs(stft)
                if magnitude.size > 0:
                    power_spectrum = magnitude**2
                    power_spectrum_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
                    spectral_entropy = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-10))
                    features['spectral_entropy'] = float(spectral_entropy)
                else:
                    features['spectral_entropy'] = 0.0
            except:
                features['spectral_entropy'] = 0.0
            
            features['lempel_ziv_complexity'] = self.lempel_ziv_complexity(y)
            features['fractal_dimension'] = self.estimate_fractal_dimension(y)
            
            features['regularity_index'] = self.regularity_index(y)
            features['predictability'] = self.predictability_measure(y)
            
        except Exception as e:
            pass
        
        return features
    
    def calculate_entropy(self, data):
        try:
            data_discrete = np.round(data * 1000).astype(int)
            _, counts = np.unique(data_discrete, return_counts=True)
            probabilities = counts / len(data_discrete)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return float(entropy)
        except:
            return 0.0
    
    def sample_entropy(self, data, m=2, r=None):
        try:
            if r is None:
                r = 0.2 * np.std(data)
            
            N = len(data)
            if N < m + 1:
                return 0.0
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template_i = patterns[i]
                    for j in range(N - m + 1):
                        if i != j and _maxdist(template_i, patterns[j], m) <= r:
                            C[i] += 1
                
                phi = np.mean(C) / (N - m)
                return phi
            
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            
            if phi_m == 0 or phi_m1 == 0:
                return 0.0
            
            return float(-np.log(phi_m1 / phi_m))
        except:
            return 0.0
    
    def approximate_entropy(self, data, m=2, r=None):
        try:
            if r is None:
                r = 0.2 * np.std(data)
            
            N = len(data)
            if N < m + 1:
                return 0.0
            
            def _phi(m):
                patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template_i = patterns[i]
                    matches = 0
                    for j in range(N - m + 1):
                        if max(abs(template_i - patterns[j])) <= r:
                            matches += 1
                    C[i] = matches / (N - m + 1)
                
                phi = np.mean(np.log(C + 1e-10))
                return phi
            
            return float(_phi(m) - _phi(m + 1))
        except:
            return 0.0
    
    def lempel_ziv_complexity(self, data):
        try:
            data_binary = (data > np.median(data)).astype(int)
            s = ''.join(map(str, data_binary))
            
            i, c = 0, 1
            n = len(s)
            
            while i + c <= n:
                substr = s[i:i + c]
                if substr in s[:i]:
                    c += 1
                else:
                    i += c
                    c = 1
            
            return float(i) / n
        except:
            return 0.0
    
    def estimate_fractal_dimension(self, data):
        try:
            N = len(data)
            if N < 10:
                return 0.0
            
            k_max = min(10, N // 4)
            L = []
            
            for k in range(1, k_max + 1):
                Lk = []
                for m in range(k):
                    Lkm = 0
                    for i in range(1, (N - m) // k):
                        Lkm += abs(data[m + i * k] - data[m + (i - 1) * k])
                    Lkm = Lkm * (N - 1) / ((N - m) // k * k)
                    Lkm = Lkm / k
                    Lkm = Lkm / k
                    Lkm = Lkm / k
                    Lkm = Lkm / k
                    Lkm = Lkm / k
                    Lkm = Lkm / k
                    Lkm = Lkm / k
                    Lkm = Lkm / k
                    if Lkm > 0:
                        Lkm = Lkm
                L.append(np.log(np.mean(Lkm)) if Lkm > 0 else 0)
            
            if len(L) > 1:
                k_vals = np.log(range(1, len(L) + 1))
                slope, _ = np.polyfit(k_vals, L, 1)
                return float(-slope)
            else:
                return 1.0
        except:
            return 1.0
    
    def regularity_index(self, data):
        try:
            if len(data) < 2:
                return 0.0
            
            diffs = np.diff(data)
            return float(1 / (1 + np.std(diffs)))
        except:
            return 0.0
    
    def predictability_measure(self, data):
        try:
            if len(data) < 3:
                return 0.0
            
            x = np.arange(len(data))
            y = data
            
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * x + intercept
            
            ss_res = np.sum((y - predicted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            if ss_tot == 0:
                return 1.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return float(max(0, r_squared))
        except:
            return 0.0
    
    def clean_features(self, features):
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
