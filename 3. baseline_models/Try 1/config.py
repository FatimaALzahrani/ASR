class Config:
    DEFAULT_DATA_PATH = None
    
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION = 3.0
    
    MIN_WORD_SAMPLES = 3
    
    KNN_NEIGHBORS = 3
    CV_FOLDS = 3
    
    MFCC_COEFFICIENTS = 13
    
    FEATURE_SIZE = 22
    
    OUTPUT_REPORT = "evaluation_report.json"
    
    PROGRESS_INTERVAL = 100