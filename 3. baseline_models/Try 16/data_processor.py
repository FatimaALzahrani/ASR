import numpy as np
import pandas as pd
from pathlib import Path
from feature_extractor import FeatureExtractor
from utils import get_speaker_from_filename
from config import EXCLUDED_WORDS, BALANCING_CONFIG

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.feature_extractor = FeatureExtractor()
    
    def load_and_prepare_data(self):
        print("Loading data with feature extraction...")
        
        data = []
        
        for word_folder in self.data_path.iterdir():
            if not word_folder.is_dir():
                continue
                
            word = word_folder.name
            if word in EXCLUDED_WORDS:
                print(f"Excluding word: {word}")
                continue
                
            print(f"Processing word: {word}")
            
            for audio_file in word_folder.glob("*.wav"):
                speaker = get_speaker_from_filename(audio_file.name)
                
                try:
                    features = self.feature_extractor.extract_features(audio_file)
                    if features is not None:
                        data.append({
                            'file_path': str(audio_file),
                            'word': word,
                            'speaker': speaker,
                            **features
                        })
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        df = pd.DataFrame(data)
        print(f"Dataset loaded: {len(df)} samples with {len(df.columns)-3} features")
        
        return df
    
    def apply_balancing(self, df, target_min=None, target_max=None):
        if target_min is None:
            target_min = BALANCING_CONFIG['target_min']
        if target_max is None:
            target_max = BALANCING_CONFIG['target_max']
            
        print("Applying smart balancing...")
        
        word_counts = df['word'].value_counts().to_dict()
        balanced_data = []
        
        for word in word_counts.keys():
            word_data = df[df['word'] == word].copy()
            current_count = len(word_data)
            
            if current_count > target_max:
                final_word_data = self._reduce_samples(word_data, target_max)
            elif current_count < target_min:
                final_word_data = self._augment_samples(word_data, target_min)
            else:
                final_word_data = word_data
            
            balanced_data.append(final_word_data)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        print(f"Balanced: {len(df)} → {len(balanced_df)} samples")
        
        return balanced_df
    
    def _reduce_samples(self, word_data, target_max):
        speakers_available = word_data['speaker'].unique()
        samples_per_speaker = max(4, target_max // len(speakers_available))
        
        reduced_data = []
        for speaker in speakers_available:
            speaker_samples = word_data[word_data['speaker'] == speaker]
            if len(speaker_samples) > samples_per_speaker:
                feature_cols = [col for col in speaker_samples.columns if col not in ['file_path', 'word', 'speaker']]
                feature_variance = speaker_samples[feature_cols].var(axis=1)
                diverse_indices = feature_variance.nlargest(samples_per_speaker).index
                sampled = speaker_samples.loc[diverse_indices]
            else:
                sampled = speaker_samples
            reduced_data.append(sampled)
        
        return pd.concat(reduced_data, ignore_index=True)
    
    def _augment_samples(self, word_data, target_min):
        augmented_data = [word_data]
        needed_samples = target_min - len(word_data)
        
        feature_cols = [col for col in word_data.columns if col not in ['file_path', 'word', 'speaker']]
        
        for i in range(needed_samples):
            original_idx = np.random.randint(0, len(word_data))
            original_sample = word_data.iloc[original_idx].copy()
            
            original_features = original_sample[feature_cols].values
            
            aug_method = i % 4
            
            if aug_method == 0:
                noise_factor = 0.03
                noise = np.random.normal(0, noise_factor * np.std(original_features), len(original_features))
                augmented_features = original_features + noise
            elif aug_method == 1:
                dropout_rate = 0.1
                mask = np.random.random(len(original_features)) > dropout_rate
                augmented_features = original_features * mask
            elif aug_method == 2:
                scale_factor = np.random.uniform(0.95, 1.05)
                augmented_features = original_features * scale_factor
            else:
                noise = np.random.normal(0, 0.02 * np.std(original_features), len(original_features))
                scale = np.random.uniform(0.98, 1.02)
                augmented_features = (original_features * scale) + noise
            
            new_sample = original_sample.copy()
            new_sample[feature_cols] = augmented_features
            new_sample['file_path'] = f"{original_sample['file_path']}_aug_{i}"
            
            augmented_data.append(pd.DataFrame([new_sample]))
        
        return pd.concat(augmented_data, ignore_index=True)
