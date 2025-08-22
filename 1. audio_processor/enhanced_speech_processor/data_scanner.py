import pandas as pd
import librosa
from pathlib import Path
from typing import List, Dict, Tuple
from audio_analyzer import AudioQualityAnalyzer
import config


class DataScanner:
    
    def __init__(self, data_root: str = config.DATA_ROOT):
        self.data_root = Path(data_root)
        self.clean_audio_dir = self.data_root / config.CLEAN_AUDIO_DIR
        self.audio_analyzer = AudioQualityAnalyzer()
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'excluded_files': 0,
            'failed_files': 0
        }
    
    def get_speaker_from_filename(self, filename: str) -> str:
        try:
            file_num = int(filename.split('.')[0])
            for range_obj, speaker in config.SPEAKER_RANGES.items():
                if file_num in range_obj:
                    return speaker
            return 'unknown'
        except:
            return 'unknown'
    
    def scan_audio_files(self) -> Tuple[pd.DataFrame, List[Dict]]:
        print("Starting comprehensive audio file scan...")
        
        audio_data = []
        failed_files = []
        
        for word_folder in self.clean_audio_dir.iterdir():
            if not word_folder.is_dir():
                continue
                
            word_name = word_folder.name
            print(f"Processing word: {word_name}")
            
            # Skip excluded words
            if word_name in config.EXCLUDED_WORDS:
                print(f"Skipping excluded word: {word_name}")
                continue
            
            audio_files = list(word_folder.glob("*.wav"))
            
            for audio_file in audio_files:
                self.stats['total_files'] += 1
                
                try:
                    # Load and analyze file
                    audio, sr = librosa.load(audio_file, sr=None)
                    duration = len(audio) / sr
                    
                    # Duration validation
                    if not (config.MIN_DURATION <= duration <= config.MAX_DURATION):
                        failed_files.append({
                            'file': str(audio_file),
                            'reason': f'Invalid duration: {duration:.2f}s'
                        })
                        self.stats['excluded_files'] += 1
                        continue
                    
                    # Speaker identification
                    speaker = self.get_speaker_from_filename(audio_file.name)
                    if speaker == 'unknown':
                        failed_files.append({
                            'file': str(audio_file),
                            'reason': 'Cannot identify speaker'
                        })
                        self.stats['excluded_files'] += 1
                        continue
                    
                    # Audio quality analysis
                    quality_metrics = self.audio_analyzer.analyze_audio_quality(audio, sr)
                    
                    # Speaker information
                    speaker_data = config.SPEAKER_INFO.get(speaker, {})
                    
                    audio_data.append({
                        'filename': audio_file.name,
                        'word': word_name,
                        'speaker': speaker,
                        'duration': duration,
                        'original_sr': sr,
                        'file_path': str(audio_file),
                        'age': speaker_data.get('age', 'unknown'),
                        'gender': speaker_data.get('gender', 'unknown'),
                        'iq_level': speaker_data.get('iq_level', 'unknown'),
                        'speech_quality': speaker_data.get('speech_quality', 'unknown'),
                        **quality_metrics
                    })
                    
                    self.stats['processed_files'] += 1
                    
                except Exception as e:
                    failed_files.append({
                        'file': str(audio_file),
                        'reason': f'Loading error: {str(e)}'
                    })
                    self.stats['failed_files'] += 1
        
        # Create DataFrame
        df = pd.DataFrame(audio_data)
        
        print(f"Scan completed:")
        print(f"  Total files: {self.stats['total_files']}")
        print(f"  Processed: {self.stats['processed_files']}")
        print(f"  Excluded: {self.stats['excluded_files']}")
        print(f"  Failed: {self.stats['failed_files']}")
        
        return df, failed_files
    
    def get_processing_stats(self) -> Dict:
        return self.stats.copy()