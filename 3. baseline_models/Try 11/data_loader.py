import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from config import Config


class ComprehensiveDataLoader:
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.speaker_mapping = Config.SPEAKER_MAPPING
        self.speaker_profiles = Config.SPEAKER_PROFILES
        
    def get_speaker_from_filename(self, filename):
        clean_filename = filename.replace("-", "").replace("_processed", "")
        
        try:
            import re
            numbers = re.findall(r'\d+', clean_filename)
            
            if numbers:
                file_num = int(numbers[0])
                
                for num_range, speaker in self.speaker_mapping.items():
                    if file_num in num_range:
                        return speaker
                        
        except Exception as e:
            print(f"Error extracting speaker from {filename}: {e}")
        
        for speaker in self.speaker_profiles.keys():
            if speaker in filename:
                return speaker
                
        return "غير معروف"
    
    def scan_audio_files(self):
        print("Scanning audio files...")
        
        dataset = []
        word_counts = defaultdict(int)
        
        for word_folder in self.data_path.iterdir():
            if not word_folder.is_dir():
                continue
                
            word = word_folder.name
            print(f"Processing word: {word}")
            
            audio_extensions = Config.AUDIO_EXTENSIONS
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(word_folder.glob(f"*{ext}"))
                audio_files.extend(word_folder.glob(f"**/*{ext}"))
            
            files_processed = 0
            for audio_file in audio_files:
                if not audio_file.exists():
                    continue
                    
                speaker = self.get_speaker_from_filename(audio_file.name)
                
                if speaker != "غير معروف":
                    dataset.append({
                        'file_path': str(audio_file),
                        'word': word,
                        'speaker': speaker,
                        'filename': audio_file.name,
                        **self.speaker_profiles.get(speaker, {})
                    })
                    
                    word_counts[word] += 1
                    files_processed += 1
            
            print(f"Processed {files_processed} files")
        
        df = pd.DataFrame(dataset)
        
        print(f"Total files: {len(df)}")
        print(f"Total words: {len(word_counts)}")
        print(f"Total speakers: {df['speaker'].nunique()}")
        
        if len(df) == 0:
            print("No audio files found!")
            print(f"Check path: {self.data_path}")
            return None
        
        return df, word_counts