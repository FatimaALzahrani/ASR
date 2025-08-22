from config import Config

class SpeakerProfileManager:
    def __init__(self):
        self.speaker_profiles = Config.SPEAKER_PROFILES
        self.word_difficulty = Config.WORD_DIFFICULTY
    
    def get_speaker_profile(self, filename):
        try:
            file_num = int(filename.split('.')[0])
            for num_range, profile in self.speaker_profiles.items():
                if file_num in num_range:
                    return profile
        except:
            pass
        return {
            "name": "Unknown", "quality": "متوسط", "clarity": 0.50, 
            "strategy": "focus_common", "target_words": 15, "augment_factor": 2.0
        }
    
    def get_word_difficulty(self, word):
        for difficulty, words in self.word_difficulty.items():
            if word in words:
                return difficulty
        if len(word) <= 3:
            return "easy"
        elif len(word) <= 6:
            return "medium"
        else:
            return "hard"