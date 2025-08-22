import re


class SpeakerIdentifier:
    def __init__(self):
        self.speaker_mapping = {
            'Ahmed': list(range(0, 7)),
            'Asem': list(range(7, 14)), 
            'Haifa': list(range(14, 21)),
            'Aseel': list(range(21, 29)),
            'Wessam': list(range(29, 37))
        }
    
    def extract_file_number(self, filename):
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[0])
        return -1
    
    def get_speaker_from_number(self, file_number):
        for speaker, numbers in self.speaker_mapping.items():
            if file_number in numbers:
                return speaker
        return "Unknown"