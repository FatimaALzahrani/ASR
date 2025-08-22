SPEAKER_PROFILES = {
    range(0, 7): {
        "name": "Ahmed", "quality": "medium", "clarity": 0.65, 
        "min_samples": 20, "preferred_models": ["RandomForest", "ExtraTrees"]
    },
    range(7, 14): {
        "name": "Asem", "quality": "high", "clarity": 0.85, 
        "min_samples": 15, "preferred_models": ["ExtraTrees", "XGBoost"]
    },
    range(14, 21): {
        "name": "Haifa", "quality": "medium", "clarity": 0.70, 
        "min_samples": 20, "preferred_models": ["RandomForest", "LightGBM"]
    },
    range(21, 29): {
        "name": "Aseel", "quality": "low", "clarity": 0.45, 
        "min_samples": 25, "preferred_models": ["RandomForest", "LogisticRegression"]
    },
    range(29, 37): {
        "name": "Wessam", "quality": "medium-high", "clarity": 0.75, 
        "min_samples": 18, "preferred_models": ["ExtraTrees", "XGBoost"]
    }
}

EXCLUDED_WORDS = {"نوم"}

MAX_DURATION = 3.0
DEFAULT_SR = 22050

MFCC_CONFIGS = [
    {'n_mfcc': 13, 'n_fft': 2048, 'hop_length': 512},
    {'n_mfcc': 20, 'n_fft': 1024, 'hop_length': 256}
]

F0_CONFIGS = [
    {'fmin': 80, 'fmax': 400},
    {'fmin': 100, 'fmax': 300}
]

QUALITY_MAPPING = {
    "high": 0.9, 
    "medium-high": 0.75, 
    "medium": 0.6, 
    "low": 0.4
}
