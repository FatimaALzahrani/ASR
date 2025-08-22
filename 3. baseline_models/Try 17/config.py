from pathlib import Path

class Config:
    DEFAULT_DATA_PATH = "C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Codes/Real Codes/01_data_processing2/data/clean"
    DEFAULT_OUTPUT_PATH = "enhanced_accuracy_results"
    
    SPEAKER_PROFILES = {
        range(0, 7): {
            "name": "Ahmed", "quality": "متوسط", "clarity": 0.65, 
            "strategy": "focus_common", "target_words": 25, "augment_factor": 2.0
        },
        range(7, 14): {
            "name": "Asem", "quality": "عالي", "clarity": 0.85, 
            "strategy": "maximize_diversity", "target_words": 15, "augment_factor": 1.5
        },
        range(14, 21): {
            "name": "Haifa", "quality": "متوسط", "clarity": 0.70, 
            "strategy": "focus_common", "target_words": 20, "augment_factor": 2.5
        },
        range(21, 29): {
            "name": "Aseel", "quality": "منخفض", "clarity": 0.45, 
            "strategy": "focus_easy", "target_words": 20, "augment_factor": 3.0
        },
        range(29, 37): {
            "name": "Wessam", "quality": "متوسط-عالي", "clarity": 0.75, 
            "strategy": "balanced", "target_words": 18, "augment_factor": 1.8
        }
    }
    
    WORD_DIFFICULTY = {
        "easy": ["بابا", "ماما", "موز", "ماء", "نام", "جا", "راح", "يد", "عين", "فم"],
        "medium": ["بيت", "كتاب", "قلم", "كاس", "كرسي", "باب", "سرير", "تفاح", "حليب", "خبز"],
        "hard": ["مستشفى", "مدرسة", "استاذة", "السلام عليكم", "تفاحة", "مفتاح", "صاروخ"]
    }
    
    EXCLUDED_WORDS = {"نوم"}
    
    QUALITY_MAPPING = {"عالي": 0.9, "متوسط-عالي": 0.75, "متوسط": 0.6, "منخفض": 0.4}