import os
import re
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor


class DataManager:
    def __init__(self, sample_rate=22050, min_samples_per_word=3, min_quality_score=0.3, random_state=42):
        self.sample_rate = sample_rate
        self.min_samples_per_word = min_samples_per_word
        self.min_quality_score = min_quality_score
        self.random_state = random_state
        
        self.speaker_mapping = {
            'أحمد': list(range(0, 7)),
            'عاصم': list(range(7, 14)),
            'هيفاء': list(range(14, 21)),
            'أسيل': list(range(21, 29)),
            'وسام': list(range(29, 37))
        }
        
        self.number_to_speaker = {}
        for speaker, numbers in self.speaker_mapping.items():
            for num in numbers:
                self.number_to_speaker[num] = speaker
        
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor()
    
    def extract_speaker_from_filename(self, filename):
        name = os.path.splitext(filename)[0]
        
        number_match = re.search(r'(\d+)', name)
        if number_match:
            file_number = int(number_match.group(1))
            return self.number_to_speaker.get(file_number, 'unknown')
        
        return 'unknown'
    
    def load_data_from_folders(self, data_path):
        data_records = []
        
        print(f"Loading data from: {data_path}")
        
        for word_folder in os.listdir(data_path):
            word_path = os.path.join(data_path, word_folder)
            
            if not os.path.isdir(word_path):
                continue
                
            print(f"Processing word: {word_folder}")
            
            for audio_file in os.listdir(word_path):
                if audio_file.endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    file_path = os.path.join(word_path, audio_file)
                    
                    speaker = self.extract_speaker_from_filename(audio_file)
                    
                    if speaker != 'unknown':
                        data_records.append({
                            'file_path': file_path,
                            'word': word_folder,
                            'speaker': speaker,
                            'filename': audio_file
                        })
        
        df = pd.DataFrame(data_records)
        
        print(f"Total samples: {len(df)}")
        print(f"Unique words: {df['word'].nunique()}")
        print(f"Unique speakers: {df['speaker'].nunique()}")
        print(f"Speaker distribution:")
        for speaker, count in df['speaker'].value_counts().items():
            print(f"  {speaker}: {count} samples")
        
        return df
    
    def filter_by_sample_count(self, df):
        print(f"Filtering words with less than {self.min_samples_per_word} samples")
        
        word_counts = df['word'].value_counts()
        valid_words = word_counts[word_counts >= self.min_samples_per_word].index
        filtered_df = df[df['word'].isin(valid_words)]
        
        removed_words = set(df['word'].unique()) - set(valid_words)
        
        print(f"Words removed: {len(removed_words)}")
        for word in removed_words:
            count = word_counts[word]
            print(f"  {word}: {count} samples")
        
        print(f"Remaining words: {len(valid_words)}")
        print(f"Remaining samples: {len(filtered_df)}")
        
        return filtered_df
    
    def filter_by_quality(self, df):
        print(f"Filtering samples by quality (minimum: {self.min_quality_score})")
        
        valid_rows = []
        processed_count = 0
        
        for idx, row in df.iterrows():
            try:
                y, sr = librosa.load(row['file_path'], sr=self.sample_rate)
                
                quality_score = self.feature_extractor.quality_analyzer.calculate_audio_quality_score(y, sr)
                
                row_with_quality = row.copy()
                row_with_quality['quality_score'] = quality_score
                
                if quality_score >= self.min_quality_score:
                    valid_rows.append(row_with_quality)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count}/{len(df)} files")
                    
            except Exception as e:
                print(f"  Failed to process {row['filename']}: {str(e)}")
        
        if valid_rows:
            filtered_df = pd.DataFrame(valid_rows)
            filtered_df = filtered_df.reset_index(drop=True)
        else:
            filtered_df = df.iloc[0:0].copy()
            filtered_df['quality_score'] = []
        
        print(f"Total processed: {len(df)}")
        print(f"Samples meeting quality threshold: {len(filtered_df)}")
        print(f"Samples removed: {len(df) - len(filtered_df)}")
        
        return filtered_df
    
    def create_speaker_independent_split(self, df, test_size=0.3):
        print("Creating speaker-independent train/test split")
        
        speakers = df['speaker'].unique()
        print(f"Total speakers: {len(speakers)}")
        print(f"Speakers: {list(speakers)}")
        
        train_speakers, test_speakers = train_test_split(
            speakers, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        print(f"Train speakers: {list(train_speakers)}")
        print(f"Test speakers: {list(test_speakers)}")
        
        train_df = df[df['speaker'].isin(train_speakers)]
        test_df = df[df['speaker'].isin(test_speakers)]
        
        train_words = set(train_df['word'].unique())
        test_words = set(test_df['word'].unique())
        common_words = train_words.intersection(test_words)
        
        print(f"Words in train: {len(train_words)}")
        print(f"Words in test: {len(test_words)}")
        print(f"Common words: {len(common_words)}")
        
        train_df_filtered = train_df[train_df['word'].isin(common_words)]
        test_df_filtered = test_df[test_df['word'].isin(common_words)]
        
        print(f"Final train size: {len(train_df_filtered)}")
        print(f"Final test size: {len(test_df_filtered)}")
        
        return train_df_filtered, test_df_filtered
    
    def process_audio_file(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            y_enhanced = self.audio_processor.enhance_audio_signal(y, sr)
            features = self.feature_extractor.extract_comprehensive_features(y_enhanced, sr)
            return features
        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")
            return None
    
    def generate_dataset_statistics(self, full_df, train_df, test_df):
        feature_columns = [col for col in full_df.columns 
                          if col not in ['file_path', 'word', 'speaker', 'filename']]
        
        stats = {
            'total_samples': len(full_df),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'unique_words': full_df['word'].nunique(),
            'unique_speakers': full_df['speaker'].nunique(),
            'feature_count': len(feature_columns),
            'word_distribution': full_df['word'].value_counts().to_dict(),
            'speaker_distribution': full_df['speaker'].value_counts().to_dict(),
            'train_speaker_distribution': train_df['speaker'].value_counts().to_dict(),
            'test_speaker_distribution': test_df['speaker'].value_counts().to_dict(),
            'quality_stats': {
                'mean_quality': full_df['quality_score'].mean(),
                'min_quality': full_df['quality_score'].min(),
                'max_quality': full_df['quality_score'].max(),
                'std_quality': full_df['quality_score'].std()
            }
        }
        
        return stats
