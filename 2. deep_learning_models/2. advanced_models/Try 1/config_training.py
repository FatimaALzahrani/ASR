import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
DATA_PATHS = {
    'train': "data/processed/train.csv",
    'validation': "data/processed/validation.csv", 
    'test': "data/processed/test.csv",
    'enhanced': "data/enhanced",
    'clean': "data/clean"
}

# Audio processing
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'max_length': 100,
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 160,
    'n_mels': 80,
    'f_min': 80,
    'f_max': 8000
}

# Training parameters
TRAINING_CONFIG = {
    'batch_sizes': {
        'hmm_dnn': 32,
        'rnn_cnn': 32,
        'end_to_end': 24
    },
    'learning_rates': {
        'hmm_dnn': 0.001,
        'rnn_cnn': 0.001,
        'end_to_end': 0.001
    },
    'epochs': {
        'hmm_dnn': 30,
        'rnn_cnn': 35,
        'end_to_end': 25
    },
    'weight_decay': 1e-4,
    'max_patience': 10,
    'scheduler_step': 10,
    'scheduler_gamma': 0.7,
    'grad_clip_norm': 1.0
}

# Model architectures
MODEL_CONFIG = {
    'hmm_dnn': {
        'input_dim': 39,
        'hidden_dim': 512,
        'feature_type': 'mfcc'
    },
    'rnn_cnn': {
        'input_dim': 17,
        'hidden_dim': 256,
        'feature_type': 'combined'
    },
    'end_to_end': {
        'input_dim': 80,
        'feature_type': 'mel_spectrogram'
    }
}

# Output files
OUTPUT_CONFIG = {
    'model_files': {
        'hmm_dnn': 'fixed_hmm_dnn_best.pth',
        'rnn_cnn': 'fixed_rnn_cnn_best.pth',
        'end_to_end': 'fixed_end_to_end_best.pth'
    },
    'results_file': 'fixed_training_results.json',
    'models_dir': 'models'
}