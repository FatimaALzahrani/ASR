#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random

class DataAugmentation:
    
    def apply_intelligent_augmentation(self, df, speaker_profile):
        augment_factor = speaker_profile.get("augment_factor", 2.0)
        speaker_quality = speaker_profile.get("overall_quality", "متوسط")
        
        if augment_factor <= 1.0:
            return df
        
        print(f"Applying data augmentation (factor: {augment_factor})")
        
        non_feature_cols = [
            'file_path', 'word', 'speaker', 'name', 'age', 'gender', 'iq',
            'overall_quality', 'clarity', 'strategy', 'target_words', 'augment_factor'
        ]
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        augmented_data = [df]
        
        current_count = len(df)
        target_count = int(current_count * augment_factor)
        needed_samples = target_count - current_count
        
        if needed_samples <= 0:
            return df
        
        print(f"Original samples: {current_count}, Target: {target_count}, Needed: {needed_samples}")
        
        try:
            for i in range(needed_samples):
                sample_weights = self.calculate_augmentation_weights(df)
                original_idx = np.random.choice(len(df), p=sample_weights)
                original_sample = df.iloc[original_idx].copy()
                
                augmentation_type = self.select_augmentation_type(original_sample, speaker_quality, i)
                
                if augmentation_type == "gaussian_noise":
                    original_sample = self.apply_gaussian_noise(original_sample, feature_cols)
                elif augmentation_type == "scaling":
                    original_sample = self.apply_scaling(original_sample, feature_cols)
                elif augmentation_type == "shift":
                    original_sample = self.apply_shift(original_sample, feature_cols)
                elif augmentation_type == "rotation":
                    original_sample = self.apply_rotation(original_sample, feature_cols)
                elif augmentation_type == "combination":
                    original_sample = self.apply_combination(original_sample, feature_cols)
                elif augmentation_type == "adaptive":
                    original_sample = self.apply_adaptive_augmentation(original_sample, feature_cols, speaker_quality)
                
                original_sample['file_path'] = f"{original_sample['file_path']}_aug_{augmentation_type}_{i}"
                
                augmented_data.append(pd.DataFrame([original_sample]))
        
        except Exception as e:
            print(f"Augmentation error: {e}")
            return df
        
        try:
            result_df = pd.concat(augmented_data, ignore_index=True)
            print(f"Augmentation complete: {len(df)} → {len(result_df)} samples")
            
            return result_df
        except Exception as e:
            print(f"Data concatenation error: {e}")
            return df
    
    def calculate_augmentation_weights(self, df):
        try:
            weights = np.ones(len(df))
            
            for i, row in df.iterrows():
                difficulty_score = row.get('word_difficulty_score', 0.5)
                quality_score = row.get('word_quality_score', 0.5)
                
                if difficulty_score > 0.7:
                    weights[i] *= 1.5
                elif difficulty_score > 0.5:
                    weights[i] *= 1.2
                
                if quality_score < 0.5:
                    weights[i] *= 1.3
            
            weights = weights / np.sum(weights)
            return weights
            
        except:
            return np.ones(len(df)) / len(df)
    
    def select_augmentation_type(self, sample, speaker_quality, iteration):
        try:
            difficulty_score = sample.get('word_difficulty_score', 0.5)
            quality_score = sample.get('word_quality_score', 0.5)
            
            if speaker_quality == "ممتاز":
                types = ["gaussian_noise", "scaling", "rotation", "adaptive"]
            elif speaker_quality == "ضعيف":
                types = ["gaussian_noise", "combination", "adaptive"]
            else:
                types = ["gaussian_noise", "scaling", "shift", "combination"]
            
            base_type = types[iteration % len(types)]
            
            if difficulty_score > 0.7 and np.random.random() < 0.3:
                return "adaptive"
            elif quality_score < 0.5 and np.random.random() < 0.4:
                return "combination"
            else:
                return base_type
                
        except:
            return "gaussian_noise"
    
    def apply_gaussian_noise(self, sample, feature_cols):
        noise_factor = np.random.uniform(0.01, 0.03)
        for col in feature_cols:
            if col in sample and isinstance(sample[col], (int, float)):
                noise = np.random.normal(0, noise_factor * abs(sample[col]))
                sample[col] += noise
        return sample
    
    def apply_scaling(self, sample, feature_cols):
        scale_factor = np.random.uniform(0.92, 1.08)
        for col in feature_cols:
            if col in sample and isinstance(sample[col], (int, float)):
                sample[col] *= scale_factor
        return sample
    
    def apply_shift(self, sample, feature_cols):
        shift_factor = np.random.uniform(0.005, 0.02)
        for col in feature_cols:
            if col in sample and isinstance(sample[col], (int, float)):
                shift = np.random.uniform(-shift_factor, shift_factor)
                sample[col] += shift
        return sample
    
    def apply_rotation(self, sample, feature_cols):
        rotation_factor = np.random.uniform(0.98, 1.02)
        phase_shift = np.random.uniform(-0.1, 0.1)
        
        for col in feature_cols:
            if col in sample and isinstance(sample[col], (int, float)):
                original_value = sample[col]
                sample[col] = original_value * rotation_factor + phase_shift
        return sample
    
    def apply_combination(self, sample, feature_cols):
        noise_factor = np.random.uniform(0.005, 0.015)
        scale_factor = np.random.uniform(0.95, 1.05)
        shift_factor = np.random.uniform(0.002, 0.01)
        
        for col in feature_cols:
            if col in sample and isinstance(sample[col], (int, float)):
                noise = np.random.normal(0, noise_factor * abs(sample[col]))
                shift = np.random.uniform(-shift_factor, shift_factor)
                sample[col] = (sample[col] * scale_factor) + noise + shift
        return sample
    
    def apply_adaptive_augmentation(self, sample, feature_cols, speaker_quality):
        if speaker_quality == "ممتاز":
            noise_factor = np.random.uniform(0.005, 0.01)
            scale_factor = np.random.uniform(0.98, 1.02)
        elif speaker_quality == "ضعيف":
            noise_factor = np.random.uniform(0.02, 0.04)
            scale_factor = np.random.uniform(0.9, 1.1)
        else:
            noise_factor = np.random.uniform(0.01, 0.025)
            scale_factor = np.random.uniform(0.94, 1.06)
        
        difficulty_score = sample.get('word_difficulty_score', 0.5)
        
        for col in feature_cols:
            if col in sample and isinstance(sample[col], (int, float)):
                if 'mfcc' in col.lower():
                    noise = np.random.normal(0, noise_factor * 0.5 * abs(sample[col]))
                    sample[col] = sample[col] * scale_factor + noise
                elif 'f0' in col.lower():
                    noise = np.random.normal(0, noise_factor * 0.3 * abs(sample[col]))
                    sample[col] = sample[col] * scale_factor + noise
                elif 'spectral' in col.lower():
                    noise = np.random.normal(0, noise_factor * 0.4 * abs(sample[col]))
                    sample[col] = sample[col] * scale_factor + noise
                else:
                    noise = np.random.normal(0, noise_factor * abs(sample[col]))
                    sample[col] = sample[col] * scale_factor + noise
                
                if difficulty_score > 0.7:
                    additional_noise = np.random.normal(0, 0.005 * abs(sample[col]))
                    sample[col] += additional_noise
        
        return sample
