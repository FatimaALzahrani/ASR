import re


class SpeakerIdentifierFixed:
    def __init__(self):
        self.speaker_ranges = {
            range(0, 7): 'Ahmed',
            range(7, 14): 'Asem', 
            range(14, 21): 'Haifa',
            range(21, 29): 'Aseel',
            range(29, 37): 'Wessam'
        }
        
        self.speaker_info = {
            'Ahmed': {'age': 10, 'gender': 'male', 'iq_level': 38, 'speech_quality': 'weak'},
            'Asem': {'age': 11, 'gender': 'male', 'iq_level': 55, 'speech_quality': 'excellent'},
            'Haifa': {'age': 7, 'gender': 'female', 'iq_level': 64, 'speech_quality': 'good'},
            'Aseel': {'age': 16, 'gender': 'female', 'iq_level': 40, 'speech_quality': 'weak'},
            'Wessam': {'age': 6, 'gender': 'male', 'iq_level': 'medium', 'speech_quality': 'medium'}
        }
    
    def get_speaker_from_filename(self, filename: str) -> str:
        try:
            file_num_str = filename.split('.')[0]
            file_num = int(file_num_str)
            
            for range_obj, speaker in self.speaker_ranges.items():
                if file_num in range_obj:
                    return speaker
            
            numbers = re.findall(r'\d+', filename)
            if numbers:
                file_num = int(numbers[0])
                for range_obj, speaker in self.speaker_ranges.items():
                    if file_num in range_obj:
                        return speaker
            
            return 'Unknown'
            
        except Exception as e:
            print(f"Warning: Error identifying speaker for file {filename}: {e}")
            return 'Unknown'