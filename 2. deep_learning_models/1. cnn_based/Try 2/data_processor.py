import pandas as pd
import numpy as np
import librosa
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class FinalDataProcessor:
    def __init__(self, data_root="data", min_samples=3):
        self.data_root = Path(data_root)
        self.clean_dir = self.data_root / "C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean"
        self.processed_dir = self.data_root / "processed"
        self.min_samples = min_samples
        
        self.processed_dir.mkdir(exist_ok=True)
        
        self.speakers = {
            range(0, 7): 'Ahmed', 
            range(7, 14): 'Asem', 
            range(14, 21): 'Haifa', 
            range(21, 29): 'Aseel', 
            range(29, 37): 'Wessam'
        }
        
        self.speaker_info = {
            'Ahmed': {'quality': 'weak'}, 
            'Asem': {'quality': 'excellent'},
            'Haifa': {'quality': 'good'}, 
            'Aseel': {'quality': 'weak'},
            'Wessam': {'quality': 'medium'}
        }

    def get_speaker(self, filename):
        try:
            num = int(filename.split('.')[0])
            for r, speaker in self.speakers.items():
                if num in r:
                    return speaker
        except:
            pass
        return 'Unknown'

    def process_data(self):
        print("Processing final data...")
        
        data = []
        for word_dir in self.clean_dir.iterdir():
            if not word_dir.is_dir() or word_dir.name == 'sleep':
                continue
                
            files = list(word_dir.glob("*.wav"))
            if len(files) < self.min_samples:
                print(f"Excluding {word_dir.name}: only {len(files)} samples")
                continue
            
            for file in files:
                speaker = self.get_speaker(file.name)
                if speaker == 'Unknown':
                    continue
                    
                try:
                    audio, sr = librosa.load(file, sr=16000)
                    if len(audio) > 0:
                        data.append({
                            'filename': file.name,
                            'word': word_dir.name, 
                            'speaker': speaker,
                            'quality': self.speaker_info[speaker]['quality'],
                            'file_path': str(file),
                            'duration': len(audio) / sr
                        })
                except:
                    continue
        
        df = pd.DataFrame(data)
        print(f"Success: {len(df)} samples, {df['word'].nunique()} words, {df['speaker'].nunique()} speakers")
        
        return df