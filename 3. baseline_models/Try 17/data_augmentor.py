import numpy as np
import pandas as pd
from config import Config

class DataAugmentor:
    def __init__(self):
        self.word_difficulty = Config.WORD_DIFFICULTY
    
    def select_optimal_words(self, df, speaker_profile):
        strategy = speaker_profile.get("strategy", "focus_common")
        target_words = speaker_profile.get("target_words", 20)
        
        word_stats = df.groupby('word').agg({
            'word': 'count',
            'speaker': 'first'
        }).rename(columns={'word': 'count'})
        
        if strategy == "focus_easy":
            easy_words = [w for w in self.word_difficulty["easy"] if w in word_stats.index]
            medium_words = [w for w in self.word_difficulty["medium"] if w in word_stats.index]
            selected_words = easy_words[:target_words//2] + medium_words[:target_words//2]
            
        elif strategy == "focus_common":
            selected_words = word_stats.nlargest(target_words, 'count').index.tolist()
            
        elif strategy == "maximize_diversity":
            all_difficulties = []
            for difficulty in ["easy", "medium", "hard"]:
                words_in_difficulty = [w for w in self.word_difficulty[difficulty] if w in word_stats.index]
                all_difficulties.extend(words_in_difficulty[:target_words//3])
            selected_words = all_difficulties[:target_words]
            
        else:
            easy_words = [w for w in self.word_difficulty["easy"] if w in word_stats.index]
            common_words = word_stats.nlargest(target_words, 'count').index.tolist()
            selected_words = list(set(easy_words[:target_words//2] + common_words[:target_words//2]))
        
        if len(selected_words) < min(5, target_words//2):
            remaining_words = [w for w in word_stats.index if w not in selected_words]
            selected_words.extend(remaining_words[:target_words - len(selected_words)])
        
        print(f"Strategy {strategy}: selected {len(selected_words)} words from {len(word_stats)}")
        return selected_words
    
    def apply_smart_augmentation(self, df, speaker_profile):
        augment_factor = speaker_profile.get("augment_factor", 2.0)
        
        if augment_factor <= 1.0:
            return df
        
        print(f"Applying data augmentation with factor {augment_factor}")
        
        feature_cols = [col for col in df.columns if col not in [
            'file_path', 'word', 'speaker', 'name', 'quality', 'clarity', 'strategy', 'target_words', 'augment_factor'
        ]]
        
        augmented_data = [df]
        
        current_count = len(df)
        target_count = int(current_count * augment_factor)
        needed_samples = target_count - current_count
        
        if needed_samples <= 0:
            return df
        
        try:
            for i in range(needed_samples):
                original_idx = np.random.randint(0, len(df))
                original_sample = df.iloc[original_idx].copy()
                
                augmentation_type = i % 4
                
                if augmentation_type == 0:
                    noise_factor = 0.02
                    for col in feature_cols:
                        if col in original_sample and isinstance(original_sample[col], (int, float)):
                            noise = np.random.normal(0, noise_factor * abs(original_sample[col]))
                            original_sample[col] += noise
                
                elif augmentation_type == 1:
                    scale_factor = np.random.uniform(0.95, 1.05)
                    for col in feature_cols:
                        if col in original_sample and isinstance(original_sample[col], (int, float)):
                            original_sample[col] *= scale_factor
                
                elif augmentation_type == 2:
                    shift_factor = 0.01
                    for col in feature_cols:
                        if col in original_sample and isinstance(original_sample[col], (int, float)):
                            shift = np.random.uniform(-shift_factor, shift_factor)
                            original_sample[col] += shift
                
                else:
                    noise_factor = 0.01
                    scale_factor = np.random.uniform(0.98, 1.02)
                    for col in feature_cols:
                        if col in original_sample and isinstance(original_sample[col], (int, float)):
                            noise = np.random.normal(0, noise_factor * abs(original_sample[col]))
                            original_sample[col] = (original_sample[col] * scale_factor) + noise
                
                original_sample['file_path'] = f"{original_sample['file_path']}_aug_{i}"
                augmented_data.append(pd.DataFrame([original_sample]))
        
        except Exception as e:
            print(f"Error in data augmentation: {e}")
            return df
        
        try:
            result_df = pd.concat(augmented_data, ignore_index=True)
            print(f"Data augmentation: {len(df)} → {len(result_df)} samples")
            return result_df
        except:
            return df