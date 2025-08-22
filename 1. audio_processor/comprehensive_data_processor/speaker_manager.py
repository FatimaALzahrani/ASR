class SpeakerManager:
    def __init__(self):
        self.speaker_mapping = {
            'Ahmed': list(range(0, 7)),
            'Asem': list(range(7, 14)), 
            'Haifa': list(range(14, 21)),
            'Aseel': list(range(21, 29)),
            'Wessam': list(range(29, 37))
        }
        
        self.speaker_info = {
            'Ahmed': {'age': 12, 'intelligence_level': 'medium', 'speech_level': 'weak'},
            'Asem': {'age': 14, 'intelligence_level': 'high', 'speech_level': 'excellent'},
            'Haifa': {'age': 13, 'intelligence_level': 'medium', 'speech_level': 'medium'},
            'Aseel': {'age': 11, 'intelligence_level': 'low', 'speech_level': 'weak'},
            'Wessam': {'age': 15, 'intelligence_level': 'medium', 'speech_level': 'medium'}
        }
    
    def get_speaker_from_number(self, file_number):
        for speaker, numbers in self.speaker_mapping.items():
            if file_number in numbers:
                return speaker
        return "Unknown"