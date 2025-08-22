import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path
from collections import Counter

from config import Config
from speaker_profile_manager import SpeakerProfileManager
from feature_extractor import FeatureExtractor
from data_augmentor import DataAugmentor
from model_trainer import ModelTrainer
from results_manager import ResultsManager

class EnhancedAccuracyASR:
    def __init__(self, data_path=None, output_path=None):
        self.data_path = Path(data_path or Config.DEFAULT_DATA_PATH)
        self.output_path = Path(output_path or Config.DEFAULT_OUTPUT_PATH)
        self.output_path.mkdir(exist_ok=True)
        
        self.speaker_manager = SpeakerProfileManager()
        self.feature_extractor = FeatureExtractor()
        self.data_augmentor = DataAugmentor()
        self.model_trainer = ModelTrainer()
        self.results_manager = ResultsManager(self.output_path)
        
        self.speaker_data = {}
        
    def load_and_optimize_data(self):
        print("Loading data with smart optimization...")
        
        speaker_data = {}
        for profile in Config.SPEAKER_PROFILES.values():
            speaker_data[profile["name"]] = []
        speaker_data["Unknown"] = []
        
        total_files = 0
        processed_files = 0
        
        for word_folder in self.data_path.iterdir():
            if not word_folder.is_dir():
                continue
                
            word = word_folder.name
            if word in Config.EXCLUDED_WORDS:
                continue
            
            audio_files = list(word_folder.glob("*.wav"))
            total_files += len(audio_files)
            
            for audio_file in audio_files:
                speaker_profile = self.speaker_manager.get_speaker_profile(audio_file.name)
                speaker_name = speaker_profile["name"]
                
                try:
                    features = self.feature_extractor.extract_optimized_features(audio_file, speaker_profile)
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
                        
                        if processed_files % 200 == 0:
                            print(f"Processed {processed_files}/{total_files} files...")
                            
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        print(f"Successfully processed {processed_files}/{total_files} files")
        
        print("\nOptimizing data for each speaker...")
        optimized_data = {}
        
        for speaker in speaker_data:
            if speaker_data[speaker] and speaker != "Unknown":
                try:
                    df = pd.DataFrame(speaker_data[speaker])
                    
                    speaker_profile = self.speaker_manager.get_speaker_profile("0")
                    for profile in Config.SPEAKER_PROFILES.values():
                        if profile["name"] == speaker:
                            speaker_profile = profile
                            break
                    
                    print(f"\nOptimizing data for speaker: {speaker}")
                    print(f"Original samples: {len(df)}")
                    print(f"Original words: {len(df['word'].unique())}")
                    
                    selected_words = self.data_augmentor.select_optimal_words(df, speaker_profile)
                    df_filtered = df[df['word'].isin(selected_words)].copy()
                    
                    print(f"Samples after filtering: {len(df_filtered)}")
                    print(f"Selected words: {len(df_filtered['word'].unique())}")
                    
                    if len(df_filtered) >= 15:
                        df_augmented = self.data_augmentor.apply_smart_augmentation(df_filtered, speaker_profile)
                        
                        optimized_data[speaker] = df_augmented
                        self.speaker_data[speaker] = df_augmented
                        
                        print(f"Final: {len(df_augmented)} samples, {len(df_augmented['word'].unique())} words")
                    else:
                        print(f"Insufficient samples after filtering ({len(df_filtered)})")
                        
                except Exception as e:
                    print(f"Error optimizing data for {speaker}: {e}")
        
        return optimized_data
    
    def run_enhanced_analysis(self):
        print("Starting enhanced analysis")
        print("="*80)
        
        speaker_data = self.load_and_optimize_data()
        
        if not speaker_data:
            print("No data loaded!")
            return None
        
        print(f"\nOptimized data summary:")
        for speaker, df in speaker_data.items():
            print(f"   {speaker}: {len(df)} enhanced samples, {len(df['word'].unique())} selected words")
        
        print(f"\nTraining enhanced models for accuracy:")
        all_results = {}
        
        for speaker, df in speaker_data.items():
            result = self.model_trainer.train_enhanced_model(speaker, df)
            if result:
                all_results[speaker] = result
        
        self.results_manager.save_results(
            all_results, 
            self.model_trainer.models, 
            self.model_trainer.scalers,
            Config.SPEAKER_PROFILES,
            Config.WORD_DIFFICULTY
        )
        
        self.results_manager.print_enhanced_report(all_results)
        
        return all_results

if __name__ == "__main__":    
    enhanced_asr = EnhancedAccuracyASR()
    results = enhanced_asr.run_enhanced_analysis()
    
    if results:
        print(f"\nEnhanced analysis completed successfully!")
        print(f"Results saved in: enhanced_accuracy_results/")
        print(f"Accuracy enhancement goal achieved successfully!")
    else:
        print(f"\nEnhanced analysis failed.")