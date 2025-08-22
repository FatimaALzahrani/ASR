import numpy as np
from pathlib import Path
from collections import Counter

from speaker_identifier import SpeakerIdentifier
from feature_extractor import FeatureExtractor

class DatasetLoader:
    def __init__(self, exclude_speakers=None, min_samples_per_class=3, max_words=50):
        self.exclude_speakers = exclude_speakers or []
        self.min_samples_per_class = min_samples_per_class
        self.max_words = max_words
        self.speaker_identifier = SpeakerIdentifier()
        self.feature_extractor = FeatureExtractor()
    
    def load_dataset(self, data_dir):
        print(f"Loading enhanced data from: {data_dir}")
        
        audio_dir = Path(data_dir)
        if not audio_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        features_list = []
        labels_list = []
        speakers_list = []
        files_info = []
        
        processed_count = 0
        excluded_count = 0
        error_count = 0
        
        print(f"Calculating word distribution...")
        word_counts = Counter()
        
        for word_folder in audio_dir.iterdir():
            if word_folder.is_dir():
                word = word_folder.name
                audio_files = list(word_folder.glob("*.wav"))
                
                valid_files = 0
                for audio_file in audio_files:
                    speaker = self.speaker_identifier.get_speaker(audio_file.name)
                    if speaker and speaker not in self.exclude_speakers:
                        valid_files += 1
                
                word_counts[word] = valid_files
        
        valid_words = [word for word, count in word_counts.items() if count >= self.min_samples_per_class]
        
        if self.max_words and len(valid_words) > self.max_words:
            sorted_words = sorted(valid_words, key=lambda w: word_counts[w], reverse=True)
            selected_words = sorted_words[:self.max_words]
        else:
            selected_words = valid_words
        
        print(f"Selected {len(selected_words)} words from {len(word_counts)} total words:")
        for i, word in enumerate(selected_words[:10], 1):
            print(f"  {i}. {word}: {word_counts[word]} samples")
        if len(selected_words) > 10:
            print(f"  ... and {len(selected_words) - 10} more words")
        
        for word_folder in audio_dir.iterdir():
            if word_folder.is_dir():
                word = word_folder.name
                
                if word not in selected_words:
                    continue
                
                audio_files = list(word_folder.glob("*.wav"))
                
                for audio_file in audio_files:
                    try:
                        speaker = self.speaker_identifier.get_speaker(audio_file.name)
                        
                        if speaker is None:
                            error_count += 1
                            continue
                        
                        if speaker in self.exclude_speakers:
                            excluded_count += 1
                            continue
                        
                        features = self.feature_extractor.extract_features(str(audio_file))
                        
                        if features is not None:
                            features_list.append(features)
                            labels_list.append(word)
                            speakers_list.append(speaker)
                            files_info.append({
                                'file_path': str(audio_file),
                                'filename': audio_file.name,
                                'word': word,
                                'speaker': speaker
                            })
                            
                            processed_count += 1
                            
                            if processed_count % 200 == 0:
                                print(f"Processed {processed_count} files...")
                        else:
                            error_count += 1
                        
                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:
                            print(f"Processing error for {audio_file}: {e}")
        
        print(f"Successfully processed {processed_count} files")
        print(f"Excluded {excluded_count} files due to speakers")
        print(f"Excluded {error_count} files due to errors")
        
        if len(features_list) == 0:
            raise ValueError("No valid data found")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        speakers_array = np.array(speakers_list)
        
        print(f"Final data shape: {X.shape}")
        print(f"Number of unique words: {len(np.unique(y))}")
        print(f"Speaker distribution: {dict(Counter(speakers_array))}")
        
        word_distribution = Counter(y)
        min_samples = min(word_distribution.values())
        print(f"Minimum samples per word: {min_samples}")
        
        return X, y, speakers_array, files_info