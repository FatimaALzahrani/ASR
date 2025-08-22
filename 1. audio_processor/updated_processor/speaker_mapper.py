class SpeakerMapper:
    def __init__(self):
        self.speaker_mapping = {
            range(0, 7): 'Ahmed',
            range(7, 14): 'Asem',  
            range(14, 22): 'Haifa',
            range(22, 30): 'Aseel',
            range(30, 38): 'Wessam'
        }
        
        self.speaker_info = {
            'Ahmed': {'age': 10, 'gender': 'male', 'iq_level': 38, 'speech_quality': 'weak'},
            'Asem': {'age': 11, 'gender': 'male', 'iq_level': 55, 'speech_quality': 'excellent'},
            'Haifa': {'age': 7, 'gender': 'female', 'iq_level': 64, 'speech_quality': 'good'},
            'Aseel': {'age': 16, 'gender': 'female', 'iq_level': 40, 'speech_quality': 'weak'},
            'Wessam': {'age': 6, 'gender': 'male', 'iq_level': 'medium', 'speech_quality': 'medium'}
        }
    
    def get_speaker_from_filename(self, filename: str) -> str:
        if filename.count('-') != 0:
            filename = filename.replace("-", "")
            
        try:
            file_num = int(filename.split('.')[0])
            
            for file_range, speaker in self.speaker_mapping.items():
                if file_num in file_range:
                    return speaker
            
            print(f"Warning: Speaker not found for file {filename}, using 'Unknown'")
            return 'Unknown'
            
        except ValueError:
            print(f"Warning: Cannot extract number from filename {filename}")
            return 'Unknown'