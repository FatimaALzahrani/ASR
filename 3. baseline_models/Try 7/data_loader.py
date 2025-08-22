from config import *
from audio_processor import AudioProcessor

class DataLoader:
    def __init__(self, random_state=Config.RANDOM_STATE):
        self.random_state = random_state
        self.speaker_mapping = Config.SPEAKER_MAPPING
        
        self.number_to_speaker = {}
        for speaker, numbers in self.speaker_mapping.items():
            for num in numbers:
                self.number_to_speaker[num] = speaker
        
        self.audio_processor = AudioProcessor()
        self.global_feature_columns = None
    
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
                if audio_file.endswith(Config.AUDIO_EXTENSIONS):
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
        
        print("Speaker distribution:")
        speaker_counts = df['speaker'].value_counts()
        for speaker, count in speaker_counts.items():
            print(f"  {speaker}: {count} samples")
        
        return df
    
    def process_complete_dataset(self, data_path, min_samples_per_word=Config.MIN_SAMPLES_PER_WORD):
        print("Starting dataset processing...")
        
        df = self.load_data_from_folders(data_path)
        
        word_counts = df['word'].value_counts()
        valid_words = word_counts[word_counts >= min_samples_per_word].index
        df_filtered = df[df['word'].isin(valid_words)]
        
        print(f"Filtered dataset: {len(df_filtered)} samples")
        print(f"Valid words: {len(valid_words)}")
        
        print("Processing audio files...")
        processed_records = []
        failed_count = 0
        
        for idx, row in df_filtered.iterrows():
            print(f"Processing {len(processed_records)+1}/{len(df_filtered)}: {row['filename']}")
            
            features = self.audio_processor.process_audio_file(row['file_path'])
            
            if features is not None:
                record = {
                    'file_path': row['file_path'],
                    'word': row['word'],
                    'speaker': row['speaker'],
                    'filename': row['filename']
                }
                record.update(features)
                processed_records.append(record)
            else:
                failed_count += 1
        
        print(f"Successfully processed: {len(processed_records)} files")
        print(f"Failed to process: {failed_count} files")
        
        if len(processed_records) == 0:
            raise ValueError("No files were successfully processed.")
        
        features_df = pd.DataFrame(processed_records)
        
        self.global_feature_columns = [col for col in features_df.columns 
                                     if col not in ['file_path', 'word', 'speaker', 'filename']]
        
        print("Applying quality filtering...")
        if 'quality_score' in features_df.columns:
            high_quality_mask = features_df['quality_score'] >= Config.QUALITY_THRESHOLD
            features_df = features_df[high_quality_mask]
            print(f"Removed {(~high_quality_mask).sum()} low-quality samples")
        
        word_counts_final = features_df['word'].value_counts()
        valid_words_final = word_counts_final[word_counts_final >= min_samples_per_word].index
        features_df = features_df[features_df['word'].isin(valid_words_final)]
        
        print(f"Final dataset: {len(features_df)} samples")
        print(f"Final words: {len(valid_words_final)}")
        
        return features_df
    
    def create_robust_splits(self, df, test_size=Config.TEST_SIZE):
        print("Creating data splits...")
        
        train_records = []
        test_records = []
        
        word_groups = df.groupby('word')
        
        for word, word_df in word_groups:
            word_speakers = word_df['speaker'].unique()
            
            if len(word_df) >= 2:
                if len(word_speakers) > 1 and all(len(word_df[word_df['speaker'] == speaker]) >= 2 for speaker in word_speakers):
                    try:
                        word_train, word_test = train_test_split(
                            word_df, test_size=test_size, random_state=self.random_state,
                            stratify=word_df['speaker']
                        )
                        train_records.append(word_train)
                        test_records.append(word_test)
                    except ValueError:
                        word_train, word_test = train_test_split(
                            word_df, test_size=test_size, random_state=self.random_state
                        )
                        train_records.append(word_train)
                        test_records.append(word_test)
                else:
                    word_train, word_test = train_test_split(
                        word_df, test_size=test_size, random_state=self.random_state
                    )
                    train_records.append(word_train)
                    test_records.append(word_test)
            else:
                train_records.append(word_df)
        
        train_df = pd.concat(train_records, ignore_index=True) if train_records else pd.DataFrame()
        test_df = pd.concat(test_records, ignore_index=True) if test_records else pd.DataFrame()
        
        print(f"Train samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Train words: {train_df['word'].nunique()}")
        print(f"Test words: {test_df['word'].nunique()}")
        print(f"Common words: {len(set(train_df['word'].unique()) & set(test_df['word'].unique()))}")
        
        print("Train speaker distribution:")
        train_speaker_counts = train_df['speaker'].value_counts()
        for speaker, count in train_speaker_counts.items():
            print(f"  {speaker}: {count} samples")
        
        print("Test speaker distribution:")
        test_speaker_counts = test_df['speaker'].value_counts()
        for speaker, count in test_speaker_counts.items():
            print(f"  {speaker}: {count} samples")
        
        return train_df, test_df
    
    def prepare_consistent_features(self, df, target_column='word'):
        if self.global_feature_columns is None:
            feature_columns = [col for col in df.columns 
                              if col not in ['file_path', 'word', 'speaker', 'filename']]
        else:
            feature_columns = self.global_feature_columns
        
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        X = df[feature_columns].copy()
        y = df[target_column]
        
        X = X.fillna(0.0)
        X = X.replace([np.inf, -np.inf], 0.0)
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
        
        feature_std = X.std()
        non_constant_features = feature_std[feature_std > 1e-8].index
        X = X[non_constant_features]
        
        print(f"Prepared {X.shape[1]} features for ML")
        
        return X, y, X.columns.tolist()