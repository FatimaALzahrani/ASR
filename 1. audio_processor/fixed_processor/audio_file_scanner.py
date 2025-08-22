import pandas as pd
import librosa
from pathlib import Path
from typing import List, Dict


class AudioFileScannerFixed:
    def __init__(self, clean_audio_dir: Path, reports_dir: Path, 
                 excluded_words: List[str], min_duration: float, max_duration: float):
        self.clean_audio_dir = clean_audio_dir
        self.reports_dir = reports_dir
        self.excluded_words = excluded_words
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def scan_audio_files(self, speaker_identifier, quality_analyzer, numpy_converter) -> pd.DataFrame:
        print("Starting comprehensive audio file scan...")
        
        audio_data = []
        failed_files = []
        
        if not self.clean_audio_dir.exists():
            print(f"Error: Data directory not found: {self.clean_audio_dir}")
            return pd.DataFrame()
        
        for word_folder in self.clean_audio_dir.iterdir():
            if not word_folder.is_dir():
                continue
                
            word_name = word_folder.name
            print(f"Processing word: {word_name}")
            
            if word_name in self.excluded_words:
                print(f"Skipping excluded word: {word_name}")
                continue
            
            audio_files = list(word_folder.glob("*.wav"))
            
            if not audio_files:
                print(f"Warning: No WAV files found in folder: {word_name}")
                continue
            
            for audio_file in audio_files:
                try:
                    audio, sr = librosa.load(audio_file, sr=None)
                    
                    if len(audio) == 0:
                        failed_files.append({
                            'file': str(audio_file),
                            'reason': 'Empty file'
                        })
                        continue
                    
                    duration = len(audio) / sr
                    
                    if not (self.min_duration <= duration <= self.max_duration):
                        failed_files.append({
                            'file': str(audio_file),
                            'reason': f'Inappropriate duration: {duration:.2f}s'
                        })
                        continue
                    
                    speaker = speaker_identifier.get_speaker_from_filename(audio_file.name)
                    
                    quality_metrics = quality_analyzer.analyze_audio_quality(audio, sr)
                    
                    speaker_data = speaker_identifier.speaker_info.get(speaker, {
                        'age': 'Unknown',
                        'gender': 'Unknown',
                        'iq_level': 'Unknown',
                        'speech_quality': 'medium'
                    })
                    
                    audio_data.append({
                        'filename': audio_file.name,
                        'word': word_name,
                        'speaker': speaker,
                        'duration': float(duration),
                        'original_sr': int(sr),
                        'file_path': str(audio_file),
                        'age': speaker_data.get('age', 'Unknown'),
                        'gender': speaker_data.get('gender', 'Unknown'),
                        'iq_level': speaker_data.get('iq_level', 'Unknown'),
                        'speech_quality': speaker_data.get('speech_quality', 'medium'),
                        **numpy_converter.convert_numpy_types(quality_metrics)
                    })
                    
                except Exception as e:
                    failed_files.append({
                        'file': str(audio_file),
                        'reason': f'Loading error: {str(e)}'
                    })
        
        df = pd.DataFrame(audio_data)
        
        if df.empty:
            print("Error: No valid files found")
            return df
        
        if failed_files:
            failed_df = pd.DataFrame(failed_files)
            failed_path = self.reports_dir / 'failed_files.csv'
            failed_df.to_csv(failed_path, index=False, encoding='utf-8')
            print(f"Warning: {len(failed_files)} files failed processing (saved to: {failed_path})")
        
        print(f"Successfully loaded {len(df)} audio files")
        return df