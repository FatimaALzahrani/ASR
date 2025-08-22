#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from utils import UtilsHelper
from feature_extractor import FeatureExtractor
from data_augmentation import DataAugmentation

class DataLoader:
    
    def __init__(self, data_path, speaker_profiles, word_categories, difficulty_levels, word_quality_map):
        self.data_path = Path(data_path)
        self.speaker_profiles = speaker_profiles
        self.word_categories = word_categories
        self.difficulty_levels = difficulty_levels
        self.word_quality_map = word_quality_map
        self.utils = UtilsHelper()
        self.feature_extractor = FeatureExtractor()
        self.data_augmentation = DataAugmentation()
        self.global_stats = {}
    
    def load_comprehensive_data(self):
        print("Loading comprehensive data...")
        print("Target: 101 words, 1307 recordings, 5 speakers")
        
        speaker_data = {}
        for profile in self.speaker_profiles.values():
            speaker_data[profile["name"]] = []
        speaker_data["Unknown"] = []
        
        total_files = 0
        processed_files = 0
        words_found = set()
        
        print("\nProcessing word folders...")
        
        for word_folder in self.data_path.iterdir():
            if not word_folder.is_dir():
                continue
                
            word = word_folder.name
            words_found.add(word)
            
            audio_files = list(word_folder.glob("*.wav"))
            total_files += len(audio_files)
            
            print(f"  {word}: {len(audio_files)} files")
            
            for audio_file in audio_files:
                speaker_profile = self.utils.get_speaker_profile(audio_file.name, self.speaker_profiles)
                speaker_name = speaker_profile["name"]
                
                try:
                    features = self.feature_extractor.extract_comprehensive_features(
                        audio_file, speaker_profile, self.word_categories, 
                        self.difficulty_levels, self.word_quality_map
                    )
                    if features is not None:
                        data_point = {
                            'file_path': str(audio_file),
                            'word': word,
                            'speaker': speaker_name,
                            **speaker_profile,
                            **features
                        }
                        speaker_data[speaker_name].append(data_point)
                        processed_files += 1
                        
                        if processed_files % 100 == 0:
                            print(f"    Processed {processed_files}/{total_files} files...")
                            
                except Exception as e:
                    print(f"    Error processing {audio_file}: {e}")
                    continue
        
        print(f"\nLoading complete:")
        print(f"   Processed files: {processed_files}/{total_files}")
        print(f"   Words found: {len(words_found)}")
        print(f"   Speakers: {len([s for s in speaker_data.keys() if speaker_data[s] and s != 'Unknown'])}")
        
        self.global_stats = {
            'total_files': total_files,
            'processed_files': processed_files,
            'words_found': list(words_found),
            'speakers_with_data': [s for s in speaker_data.keys() if speaker_data[s]]
        }
        
        print(f"\nApplying intelligent data optimization...")
        optimized_data = {}
        
        for speaker in speaker_data:
            if speaker_data[speaker] and speaker != "Unknown":
                try:
                    df = pd.DataFrame(speaker_data[speaker])
                    
                    speaker_profile = self.utils.get_speaker_profile("0", self.speaker_profiles)
                    for profile in self.speaker_profiles.values():
                        if profile["name"] == speaker:
                            speaker_profile = profile
                            break
                    
                    print(f"\n{speaker} optimization:")
                    print(f"   Original samples: {len(df)}")
                    print(f"   Original words: {len(df['word'].unique())}")
                    print(f"   Quality: {speaker_profile.get('overall_quality', 'متوسط')}")
                    print(f"   IQ: {speaker_profile.get('iq', 'N/A')}")
                    
                    selected_words = self.select_optimal_words_advanced(df, speaker_profile)
                    df_filtered = df[df['word'].isin(selected_words)].copy()
                    
                    print(f"   Filtered samples: {len(df_filtered)}")
                    print(f"   Selected words: {len(df_filtered['word'].unique())}")
                    
                    if len(df_filtered) >= 20:
                        df_augmented = self.data_augmentation.apply_intelligent_augmentation(df_filtered, speaker_profile)
                        
                        optimized_data[speaker] = df_augmented
                        
                        print(f"   Final: {len(df_augmented)} samples, {len(df_augmented['word'].unique())} words")
                    else:
                        print(f"   Insufficient samples after filtering ({len(df_filtered)})")
                        
                except Exception as e:
                    print(f"   Error optimizing {speaker}: {e}")
        
        return optimized_data
    
    def select_optimal_words_advanced(self, df, speaker_profile):
        strategy = speaker_profile.get("strategy", "balanced")
        target_words = speaker_profile.get("target_words", 30)
        speaker_name = speaker_profile.get("name", "Unknown")
        
        word_stats = df.groupby('word').agg({
            'word': 'count',
            'speaker': 'first',
            'word_difficulty_score': 'first',
            'word_category_score': 'first',
            'word_quality_score': 'first'
        }).rename(columns={'word': 'count'})
        
        word_stats['difficulty'] = word_stats.index.map(
            lambda w: self.utils.get_word_difficulty_advanced(w, self.difficulty_levels)
        )
        word_stats['category'] = word_stats.index.map(
            lambda w: self.utils.get_word_category(w, self.word_categories)
        )
        
        print(f"   Word analysis for {speaker_name}:")
        print(f"     Available words: {len(word_stats)}")
        print(f"     Avg samples per word: {word_stats['count'].mean():.1f}")
        print(f"     Strategy: {strategy}")
        
        if strategy == "focus_easy_only":
            easy_words = word_stats[word_stats['difficulty'].isin(['very_easy', 'easy'])]
            selected_words = easy_words.nlargest(target_words, 'count').index.tolist()
            
        elif strategy == "focus_medium_easy":
            easy_medium_words = word_stats[word_stats['difficulty'].isin(['very_easy', 'easy', 'medium'])]
            selected_words = easy_medium_words.nlargest(target_words, 'count').index.tolist()
            
        elif strategy == "maximize_all":
            high_quality = word_stats[word_stats['word_quality_score'] >= 0.7]
            medium_quality = word_stats[word_stats['word_quality_score'] >= 0.5]
            
            selected_words = []
            selected_words.extend(high_quality.nlargest(target_words//2, 'count').index.tolist())
            
            remaining_target = target_words - len(selected_words)
            remaining_words = medium_quality[~medium_quality.index.isin(selected_words)]
            selected_words.extend(remaining_words.nlargest(remaining_target, 'count').index.tolist())
            
        elif strategy == "balanced_comprehensive":
            selected_words = []
            
            high_quality = word_stats[word_stats['word_quality_score'] >= 0.7]
            selected_words.extend(high_quality.nlargest(target_words//3, 'count').index.tolist())
            
            remaining_words = word_stats[~word_stats.index.isin(selected_words)]
            frequent_words = remaining_words.nlargest(target_words//3, 'count')
            selected_words.extend(frequent_words.index.tolist())
            
            remaining_target = target_words - len(selected_words)
            remaining_words = word_stats[~word_stats.index.isin(selected_words)]
            
            for difficulty in ['easy', 'medium', 'hard']:
                diff_words = remaining_words[remaining_words['difficulty'] == difficulty]
                if len(diff_words) > 0 and remaining_target > 0:
                    take_count = min(remaining_target//3, len(diff_words))
                    selected_words.extend(diff_words.nlargest(take_count, 'count').index.tolist())
                    remaining_target -= take_count
            
        elif strategy == "adaptive_mixed":
            speaker_quality = speaker_profile.get("overall_quality", "متوسط")
            
            if speaker_quality == "ممتاز":
                all_difficulties = word_stats.groupby('difficulty')
                selected_words = []
                for difficulty in ['easy', 'medium', 'hard', 'very_hard']:
                    if difficulty in all_difficulties.groups:
                        diff_words = all_difficulties.get_group(difficulty)
                        take_count = target_words // 4
                        selected_words.extend(diff_words.nlargest(take_count, 'count').index.tolist())
            else:
                easy_medium = word_stats[word_stats['difficulty'].isin(['very_easy', 'easy', 'medium'])]
                selected_words = easy_medium.nlargest(target_words, 'count').index.tolist()
        
        else:
            selected_words = word_stats.nlargest(target_words, 'count').index.tolist()
        
        min_required = min(10, target_words//3)
        if len(selected_words) < min_required:
            remaining_words = [w for w in word_stats.index if w not in selected_words]
            additional_needed = min_required - len(selected_words)
            selected_words.extend(remaining_words[:additional_needed])
        
        selected_stats = word_stats.loc[selected_words]
        print(f"     Selected words: {len(selected_words)}")
        print(f"     Avg samples: {selected_stats['count'].mean():.1f}")
        print(f"     Difficulty distribution: {selected_stats['difficulty'].value_counts().to_dict()}")
        print(f"     Avg pronunciation quality: {selected_stats['word_quality_score'].mean():.2f}")
        
        return selected_words
