"""
Configuration file for Speech Recognition Training Pipeline
"""

# Data Processing Configuration
DATA_CONFIG = {
    'data_root': 'data',
    'clean_dir': 'data/clean',
    'enhanced_dir': 'data/enhanced',
    'processed_dir': 'data/processed',
    'min_samples_per_word': 3,
    'excluded_words': ['sleep'],  # Words to exclude from processing
    'target_sample_rate': 16000,
}

# Speaker Configuration
SPEAKER_CONFIG = {
    'speaker_ranges': {
        'Ahmed': range(0, 7),
        'Asem': range(7, 14), 
        'Haifa': range(14, 21),
        'Aseel': range(21, 29),
        'Wessam': range(29, 37)
    },
    'speaker_info': {
        'Ahmed': {'quality': 'weak'},
        'Asem': {'quality': 'excellent'},
        'Haifa': {'quality': 'good'},
        'Aseel': {'quality': 'weak'},
        'Wessam': {'quality': 'medium'}
    }
}

# Model Configuration
MODEL_CONFIG = {
    'input_dim': 13,  # MFCC features
    'hidden_dim': 128,
    'dropout_rate': 0.3,
    'cnn_channels': [64, 128, 256],
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'patience': 5,  # For learning rate scheduler
    'device': 'auto',  # 'auto', 'cuda', or 'cpu'
}

# MFCC Feature Configuration
FEATURE_CONFIG = {
    'n_mfcc': 13,
    'n_fft': 400,
    'hop_length': 160,
    'sample_rate': 16000,
}

# Data Split Configuration
SPLIT_CONFIG = {
    'general_splits': {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
    },
    'personalized_splits': {
        'train_ratio': 0.8,
        'test_ratio': 0.2,
    }
}

# Output Configuration
OUTPUT_CONFIG = {
    'models_dir': 'models',
    'results_dir': 'results',
    'logs_dir': 'logs',
}