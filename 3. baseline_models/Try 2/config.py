class Config:
    WHISPER_MODEL_NAME = "openai/whisper-small"
    LANGUAGE = "ar"
    TASK = "transcribe"
    SAMPLE_RATE = 16000
    MAX_LENGTH = 50
    
    DATA_DIR = "processed" 
    RESULTS_DIR = "results"
    
    SAMPLE_SIZE = 50
    RANDOM_STATE = 42
    
    MIN_WORD_SAMPLES = 3
    TOP_WORDS_COUNT = 10
    PREDICTIONS_SAMPLE_SIZE = 5
    
    RESULTS_FILE = "training_results.json"
    SUMMARY_FILE = "results_summary.json"