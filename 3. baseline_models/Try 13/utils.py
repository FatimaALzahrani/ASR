#!/usr/bin/env python3

class UtilsHelper:
    
    def get_speaker_profile(self, filename, speaker_profiles):
        try:
            file_num = int(filename.split('.')[0])
            for num_range, profile in speaker_profiles.items():
                if file_num in num_range:
                    return profile
        except:
            pass
        return {
            "name": "Unknown", "age": "0", "gender": "غير محدد", "iq": 50,
            "overall_quality": "متوسط", "clarity": 0.50, 
            "strategy": "balanced", "target_words": 25, "augment_factor": 2.0
        }
    
    def get_word_difficulty_advanced(self, word, difficulty_levels):
        for difficulty, words in difficulty_levels.items():
            if word in words:
                return difficulty
        
        word_len = len(word)
        if word_len <= 2:
            return "very_easy"
        elif word_len <= 4:
            return "easy"
        elif word_len <= 6:
            return "medium"
        elif word_len <= 8:
            return "hard"
        else:
            return "very_hard"
    
    def get_word_category(self, word, word_categories):
        for category, words in word_categories.items():
            if word in words:
                return category
        return "غير_مصنف"
    
    def get_word_quality_for_speaker(self, word, speaker_name, word_quality_map):
        if speaker_name in word_quality_map:
            return word_quality_map[speaker_name].get(word, "متوسط")
        return "متوسط"
    
    def get_difficulty_score(self, difficulty):
        mapping = {
            "very_easy": 0.1,
            "easy": 0.3,
            "medium": 0.5,
            "hard": 0.7,
            "very_hard": 0.9
        }
        return mapping.get(difficulty, 0.5)
    
    def get_category_score(self, category):
        easy_categories = ["عائلة_وأشخاص", "أجزاء_الجسم", "أفعال_سهلة"]
        medium_categories = ["طعام_وشراب", "ألوان", "أرقام", "حيوانات"]
        hard_categories = ["أدوات_ومنزل", "طبيعة_وبيئة", "ملابس", "مدرسة", "مشاعر"]
        very_hard_categories = ["صعبة"]
        
        if category in easy_categories:
            return 0.2
        elif category in medium_categories:
            return 0.4
        elif category in hard_categories:
            return 0.6
        elif category in very_hard_categories:
            return 0.8
        else:
            return 0.5
