import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re

class RealisticDataLoader:
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        
        self.speaker_mapping = {
            range(0, 7): "أحمد",
            range(7, 14): "عاصم", 
            range(14, 21): "هيفاء",
            range(21, 29): "أسيل",
            range(29, 37): "وسام"
        }
        
        self.speaker_profiles = {
            "أحمد": {"age": 10, "iq": 38, "gender": "male", "severity": "moderate"},
            "عاصم": {"age": 11, "iq": 55, "gender": "male", "severity": "excellent"},
            "هيفاء": {"age": 7, "iq": 64, "gender": "female", "severity": "good"},
            "أسيل": {"age": 16, "iq": 40, "gender": "female", "severity": "weak"},
            "وسام": {"age": 6, "iq": 45, "gender": "male", "severity": "moderate"}
        }
        
    def get_speaker_from_filename(self, filename):
        clean_filename = filename.replace("-", "").replace("_processed", "")
        
        try:
            numbers = re.findall(r'\d+', clean_filename)
            
            if numbers:
                file_num = int(numbers[0])
                for num_range, speaker in self.speaker_mapping.items():
                    if file_num in num_range:
                        return speaker
        except:
            pass
        
        for speaker in self.speaker_profiles.keys():
            if speaker in filename:
                return speaker
                
        return "غير معروف"
    
    def scan_audio_files_with_sessions(self):
        print("Scanning audio files...")
        
        dataset = []
        word_counts = defaultdict(int)
        
        for word_folder in self.data_path.iterdir():
            if not word_folder.is_dir():
                continue
                
            word = word_folder.name
            print(f"Processing word: {word}")
            
            audio_extensions = [".wav", ".mp3", ".flac", ".m4a"]
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
                    session_id = f"{speaker}_{word}"
                    
                    dataset.append({
                        'file_path': str(audio_file),
                        'word': word,
                        'speaker': speaker,
                        'session_id': session_id,
                        'filename': audio_file.name,
                        **self.speaker_profiles.get(speaker, {})
                    })
                    
                    word_counts[word] += 1
                    files_processed += 1
            
            print(f"   Processed {files_processed} files")
        
        df = pd.DataFrame(dataset)
        
        print(f"\nDataset Summary:")
        print(f"   Total files: {len(df)}")
        print(f"   Unique words: {len(word_counts)}")
        print(f"   Speakers: {df['speaker'].nunique()}")
        print(f"   Sessions: {df['session_id'].nunique()}")
        
        return df, word_counts