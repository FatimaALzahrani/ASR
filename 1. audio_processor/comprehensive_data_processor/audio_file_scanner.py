import re
from pathlib import Path


class AudioFileScanner:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.audio_files = []
    
    def extract_file_number(self, filename):
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[0])
        return -1
    
    def scan_files(self, speaker_manager, audio_analyzer):
        print("Scanning audio files...")
        
        total_files = 0
        valid_files = 0
        
        for word_dir in self.data_path.iterdir():
            if word_dir.is_dir():
                word = word_dir.name
                print(f"Processing word: {word}")
                
                for audio_file in word_dir.glob("*.wav"):
                    total_files += 1
                    
                    try:
                        file_number = self.extract_file_number(audio_file.stem)
                        speaker = speaker_manager.get_speaker_from_number(file_number)
                        audio_info = audio_analyzer.analyze_file(audio_file)
                        
                        if audio_info:
                            self.audio_files.append({
                                'file_path': str(audio_file),
                                'word': word,
                                'file_number': file_number,
                                'speaker': speaker,
                                'duration': audio_info['duration'],
                                'sample_rate': audio_info['sample_rate'],
                                'channels': audio_info['channels'],
                                'file_size': audio_file.stat().st_size,
                                'quality_score': audio_analyzer.estimate_quality(audio_info)
                            })
                            valid_files += 1
                            
                    except Exception as e:
                        print(f"Error processing {audio_file}: {e}")
                        continue
        
        print(f"Scanned {total_files} files, {valid_files} valid files found")
        return valid_files