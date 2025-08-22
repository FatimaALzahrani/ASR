import pandas as pd
import librosa
from pathlib import Path


class AudioStructureScanner:
    def __init__(self, audio_root: Path, min_duration: float = 0.5, max_duration: float = 30.0):
        self.audio_root = audio_root
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def scan_audio_structure(self, speaker_mapper) -> pd.DataFrame:
        print("Scanning audio file structure...")
        
        audio_data = []
        
        for word_folder in self.audio_root.iterdir():
            if word_folder.is_dir():
                word_name = word_folder.name
                print(f"Processing word folder: {word_name}")
                
                audio_files = list(word_folder.glob("*.wav"))
                print(f"   Found {len(audio_files)} audio files")
                
                for audio_file in audio_files:
                    try:
                        audio, sr = librosa.load(audio_file, sr=None)
                        duration = len(audio) / sr
                        
                        if self.min_duration <= duration <= self.max_duration:
                            speaker = speaker_mapper.get_speaker_from_filename(audio_file.name)
                            
                            speaker_data = speaker_mapper.speaker_info.get(speaker, {
                                'age': 'Unknown',
                                'gender': 'Unknown', 
                                'iq_level': 'Unknown',
                                'speech_quality': 'Unknown'
                            })
                            
                            audio_data.append({
                                'filename': audio_file.name,
                                'word': word_name,
                                'text': word_name,
                                'speaker': speaker,
                                'quality': speaker_data['speech_quality'],
                                'age': speaker_data['age'],
                                'gender': speaker_data['gender'],
                                'iq_level': speaker_data['iq_level'],
                                'duration': duration,
                                'sample_rate': sr,
                                'audio_path': str(audio_file),
                                'word_folder': word_name
                            })
                        else:
                            print(f"Warning: Inappropriate duration for file {audio_file.name}: {duration:.2f}s")
                            
                    except Exception as e:
                        print(f"Error processing file {audio_file}: {e}")
        
        if not audio_data:
            print("No valid audio files found")
            return pd.DataFrame()
        
        df = pd.DataFrame(audio_data)
        
        print(f"\nData Statistics:")
        print(f"   Total files: {len(df)}")
        print(f"   Number of words: {df['word'].nunique()}")
        print(f"   Number of speakers: {df['speaker'].nunique()}")
        print(f"   Total duration: {df['duration'].sum():.1f} seconds")
        
        print(f"\nWord distribution:")
        for word, count in df['word'].value_counts().items():
            print(f"   {word}: {count} files")
        
        print(f"\nSpeaker distribution:")
        for speaker, count in df['speaker'].value_counts().items():
            print(f"   {speaker}: {count} files")
        
        return df