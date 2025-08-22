import os

class SpeakerIdentifier:
    def __init__(self):
        self.speakers = {
            'أحمد': range(0, 7),
            'عاصم': range(7, 14),
            'هيفاء': range(14, 21),
            'أسيل': range(21, 29),
            'وسام': range(29, 37)
        }
    
    def get_speaker(self, filename):
        try:
            name = os.path.splitext(filename)[0]
            
            if not name or name == "":
                return None
                
            if name == "0":
                number = 0
            else:
                cleaned_name = name.lstrip("0")
                if not cleaned_name:
                    number = 0
                else:
                    number = int(cleaned_name)

            for speaker, rng in self.speakers.items():
                if number in rng:
                    return speaker
            return None
            
        except (ValueError, AttributeError):
            return None