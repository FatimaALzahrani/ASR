import pandas as pd
from pathlib import Path
from collections import defaultdict
from feature_extractor import FeatureExtractor


class DataLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.feature_extractor = FeatureExtractor()
        
        self.speaker_mapping = {
            range(0, 7): "Ahmed",
            range(7, 14): "Asem", 
            range(14, 21): "Haifa",
            range(21, 29): "Aseel",
            range(29, 37): "Wessam"
        }
        
        self.excluded_words = {"نوم"}
        
    def get_speaker_from_filename(self, filename):
        filename = filename.replace("-","")
        try:
            file_num = int(filename.split('.')[0])
            for num_range, speaker in self.speaker_mapping.items():
                if file_num in num_range:
                    return speaker
        except:
            pass
        return "Unknown"
    
    def load_data(self):
        print("Loading audio data and extracting features...")
        
        data = []
        word_counts = defaultdict(int)
        speaker_word_counts = defaultdict(lambda: defaultdict(int))
        
        for word_folder in self.data_path.iterdir():
            if not word_folder.is_dir():
                continue
                
            word = word_folder.name
            if word in self.excluded_words:
                print(f"Excluding word: {word}")
                continue
                
            print(f"Processing word: {word}")
            
            word_samples = 0
            for audio_file in word_folder.glob("*.wav"):
                speaker = self.get_speaker_from_filename(audio_file.name)
                
                try:
                    features = self.feature_extractor.extract_features(audio_file)
                    if features is not None:
                        data.append({
                            'file_path': str(audio_file),
                            'word': word,
                            'speaker': speaker,
                            **features
                        })
                        word_counts[word] += 1
                        speaker_word_counts[speaker][word] += 1
                        word_samples += 1
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
            
            print(f"  Loaded {word_samples} samples for '{word}'")
        
        df = pd.DataFrame(data)
        word_frequencies = dict(word_counts)
        
        print(f"\nDataset Summary:")
        print(f"Total samples: {len(df)}")
        print(f"Total words: {len(word_counts)}")
        print(f"Total speakers: {len(speaker_word_counts)}")
        
        if len(df) == 0:
            print("ERROR: No samples loaded! Check data path.")
            return None, None, None
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 words:")
        for word, count in sorted_words[:10]:
            print(f"  {word}: {count} samples")
        
        print(f"\nSpeaker distribution:")
        for speaker, words in speaker_word_counts.items():
            total_samples = sum(words.values())
            unique_words = len(words)
            print(f"  {speaker}: {total_samples} samples, {unique_words} words")
        
        return df, word_frequencies, speaker_word_counts