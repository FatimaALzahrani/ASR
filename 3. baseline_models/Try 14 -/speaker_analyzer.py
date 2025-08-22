import os
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from feature_extractor import FeatureExtractor
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from result_manager import ResultManager
from utils import get_speaker_profile
from config import SPEAKER_PROFILES, EXCLUDED_WORDS

class SpeakerAnalyzer:
    
    def __init__(self, data_path="data/clean", output_path="results"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        
        self.feature_extractor = FeatureExtractor()
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.result_manager = ResultManager(output_path)
        
        self.speaker_profiles = SPEAKER_PROFILES
        self.speaker_data = {}
    
    def load_data(self):
        print("Loading audio data...")
        
        speaker_data = {}
        for profile in self.speaker_profiles.values():
            speaker_data[profile["name"]] = []
        speaker_data["Unknown"] = []
        
        total_files = 0
        processed_files = 0
        failed_files = 0
        
        for word_folder in self.data_path.iterdir():
            if not word_folder.is_dir():
                continue
                
            word = word_folder.name
            if word in EXCLUDED_WORDS:
                print(f"Excluded word: {word}")
                continue
            
            audio_files = list(word_folder.glob("*.wav"))
            total_files += len(audio_files)
            print(f"Processing word: {word} ({len(audio_files)} files)")
            
            for audio_file in audio_files:
                speaker_profile = get_speaker_profile(audio_file.name)
                speaker_name = speaker_profile["name"]
                
                try:
                    features = self.feature_extractor.extract_features(audio_file, speaker_profile)
                    if features is not None and len(features) > 50:
                        data_point = {
                            'file_path': str(audio_file),
                            'word': word,
                            'speaker': speaker_name,
                            **speaker_profile,
                            **features
                        }
                        speaker_data[speaker_name].append(data_point)
                        processed_files += 1
                    else:
                        failed_files += 1
                        
                    if processed_files % 100 == 0:
                        print(f"Processed {processed_files}/{total_files} files... (failed: {failed_files})")
                        
                except Exception as e:
                    print(f"Failed to process {audio_file}: {e}")
                    failed_files += 1
                    continue
        
        print(f"\nConverting to DataFrames...")
        final_speaker_data = {}
        for speaker in speaker_data:
            if speaker_data[speaker]:
                try:
                    df = pd.DataFrame(speaker_data[speaker])
                    df = df.dropna(subset=['word', 'speaker'])
                    
                    if len(df) > 0:
                        final_speaker_data[speaker] = df
                        self.speaker_data[speaker] = df
                        print(f"✓ {speaker}: {len(df)} samples, {len(df['word'].unique())} words")
                    else:
                        print(f"✗ {speaker}: No valid samples")
                except Exception as e:
                    print(f"✗ Error creating DataFrame for {speaker}: {e}")
            else:
                print(f"✗ {speaker}: No samples")
        
        print(f"\nLoading Summary:")
        print(f"   Total files: {total_files}")
        print(f"   Successfully processed: {processed_files}")
        print(f"   Failed: {failed_files}")
        print(f"   Success rate: {(processed_files/total_files)*100:.1f}%")
        
        return self.speaker_data
    
    def run_analysis(self):
        print("Starting comprehensive analysis")
        print("="*60)
        
        speaker_data = self.load_data()
        
        if not speaker_data:
            print("No data loaded!")
            return None
        
        all_results = {}
        
        print(f"\nData Summary:")
        total_samples = 0
        total_words = set()
        
        for speaker, df in speaker_data.items():
            if df is not None and len(df) > 0:
                words = df['word'].unique()
                total_samples += len(df)
                total_words.update(words)
                
                speaker_profile = get_speaker_profile("0")
                for profile in self.speaker_profiles.values():
                    if profile["name"] == speaker:
                        speaker_profile = profile
                        break
                
                quality = speaker_profile.get("quality", "unknown")
                clarity = speaker_profile.get("clarity", 0.5)
                min_samples = speaker_profile.get("min_samples", 15)
                
                status = "Ready" if len(df) >= min_samples else "Low"
                print(f"   {speaker}: {len(df)} samples, {len(words)} words, quality: {quality} ({clarity:.2f}) {status}")
        
        print(f"\nOverall Statistics:")
        print(f"   Total samples: {total_samples}")
        print(f"   Total words: {len(total_words)}")
        print(f"   Speakers: {len([s for s, df in speaker_data.items() if df is not None and len(df) > 0])}")
        
        print(f"\nTraining Models:")
        successful_trainings = 0
        
        for speaker, df in speaker_data.items():
            if df is not None and len(df) > 0:
                speaker_profile = get_speaker_profile("0")
                for profile in self.speaker_profiles.values():
                    if profile["name"] == speaker:
                        speaker_profile = profile
                        break
                
                result = self.model_trainer.train_model(speaker, df, self.data_processor, speaker_profile)
                if result:
                    all_results[speaker] = result
                    successful_trainings += 1
                else:
                    print(f"Failed to train model for {speaker}")
        
        print(f"\nTraining Results:")
        print(f"   Successful: {successful_trainings}/{len(speaker_data)} speakers")
        print(f"   Success rate: {(successful_trainings/len(speaker_data))*100:.1f}%")
        
        self.result_manager.save_results(
            all_results, 
            self.model_trainer.get_all_models(), 
            self.data_processor.speaker_scalers, 
            self.speaker_profiles
        )
        
        self.result_manager.print_report(all_results)
        
        return all_results

if __name__ == "__main__":
    print("Down Syndrome Speech Recognition System")
    print("Advanced Speaker-Adaptive ASR")
    
    analyzer = SpeakerAnalyzer()
    results = analyzer.run_analysis()
    
    if results:
        print(f"\nAnalysis completed successfully!")
        print(f"All results saved to: results/")
        print(f"System ready for deployment and research publication!")
    else:
        print(f"\nNo results obtained. Check data path.")
        print(f"Ensure .wav files exist in the specified directory.")
