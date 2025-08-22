import os
import pandas as pd
from feature_extractor import AdvancedFeatureExtractor
from audio_utils import load_audio, validate_audio_file
from file_utils import ensure_directory_exists, save_dataframe
from settings import FEATURES_PATH, PROCESSED_DATA_PATH
from tqdm import tqdm


class FeatureExtractionPipeline:
    def __init__(self, audio_dir=None, output_dir=None):
        self.audio_dir = audio_dir or PROCESSED_DATA_PATH
        self.output_dir = output_dir or FEATURES_PATH
        self.extractor = AdvancedFeatureExtractor()
        
        ensure_directory_exists(self.output_dir)
        
        print("Feature Extraction Pipeline initialized")
        print(f"Audio directory: {self.audio_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def extract_speaker_word_from_path(self, file_path):
        filename = os.path.basename(file_path)
        
        # Try different naming conventions
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 2:
                speaker = parts[0]
                word = parts[1].split('.')[0]
                return speaker, word
        
        # Fallback: use parent directory as speaker
        parent_dir = os.path.basename(os.path.dirname(file_path))
        word = os.path.splitext(filename)[0]
        
        return parent_dir, word
    
    def process_audio_file(self, file_path):
        try:
            is_valid, validation_msg = validate_audio_file(file_path)
            if not is_valid:
                print(f"Invalid audio file {file_path}: {validation_msg}")
                return None
            
            audio, sr = load_audio(file_path)
            if audio is None:
                print(f"Failed to load audio: {file_path}")
                return None
            
            features = self.extractor.extract_features(audio, sr)
            if not features:
                print(f"Failed to extract features: {file_path}")
                return None
            
            speaker, word = self.extract_speaker_word_from_path(file_path)
            
            features['file_path'] = file_path
            features['word'] = word
            features['speaker'] = speaker
            features['actual_duration'] = len(audio) / sr
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def extract_features_from_directory(self):
        print("Scanning for audio files...")
        
        audio_files = []
        for root, dirs, files in os.walk(self.audio_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print(f"No audio files found in {self.audio_dir}")
            return None
        
        print(f"Found {len(audio_files)} audio files")
        
        all_features = []
        
        print("Extracting features...")
        for file_path in tqdm(audio_files, desc="Processing files"):
            features = self.process_audio_file(file_path)
            if features:
                all_features.append(features)
        
        if not all_features:
            print("No features extracted")
            return None
        
        features_df = pd.DataFrame(all_features)
        
        print(f"Extracted features from {len(features_df)} files")
        print(f"Feature columns: {len(features_df.columns)}")
        print(f"Unique speakers: {features_df['speaker'].nunique()}")
        print(f"Unique words: {features_df['word'].nunique()}")
        
        return features_df
    
    def save_features(self, features_df):
        output_file = os.path.join(self.output_dir, "features_for_modeling.csv")
        
        success = save_dataframe(features_df, output_file, format='csv')
        
        if success:
            print(f"Features saved to: {output_file}")
            
            # Save summary
            summary = {
                'total_samples': len(features_df),
                'unique_speakers': int(features_df['speaker'].nunique()),
                'unique_words': int(features_df['word'].nunique()),
                'feature_count': int(len(features_df.columns) - 4),  # exclude metadata columns
                'speakers': features_df['speaker'].value_counts().to_dict(),
                'words': features_df['word'].value_counts().to_dict()
            }
            
            summary_file = os.path.join(self.output_dir, "extraction_summary.json")
            import json
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"Summary saved to: {summary_file}")
            return output_file
        else:
            print("Failed to save features")
            return None
    
    def run_extraction(self):
        print("Starting feature extraction pipeline...")
        
        if not os.path.exists(self.audio_dir):
            print(f"Audio directory not found: {self.audio_dir}")
            print("Please create the directory and add audio files, or specify a different path")
            return False
        
        features_df = self.extract_features_from_directory()
        
        if features_df is not None:
            output_file = self.save_features(features_df)
            if output_file:
                print("Feature extraction completed successfully!")
                return True
        
        print("Feature extraction failed!")
        return False


def create_sample_data():
    print("Creating sample data structure...")
    
    # Create directory structure for sample data
    sample_audio_dir = "sample_audio_data"
    speakers = ["أحمد", "عاصم", "هيفاء", "أسيل", "وسام"]
    words = ["ماما", "بابا", "ماء", "خبز", "حليب", "كتاب", "قلم", "شمس", "قمر", "بيت"]
    
    os.makedirs(sample_audio_dir, exist_ok=True)
    
    for speaker in speakers:
        speaker_dir = os.path.join(sample_audio_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        
        print(f"Create audio files in: {speaker_dir}")
        for word in words[:5]:  # 5 words per speaker
            sample_file = os.path.join(speaker_dir, f"{word}.wav")
            print(f"  Expected: {sample_file}")
    
    print("\nSample directory structure created!")
    print(f"Add your audio files to: {sample_audio_dir}")
    print("Then run: python extract_features.py")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from audio files')
    parser.add_argument('--audio-dir', type=str, help='Directory containing audio files')
    parser.add_argument('--output-dir', type=str, help='Output directory for features')
    parser.add_argument('--create-sample', action='store_true', help='Create sample directory structure')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
        return 0
    
    pipeline = FeatureExtractionPipeline(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir
    )
    
    success = pipeline.run_extraction()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
