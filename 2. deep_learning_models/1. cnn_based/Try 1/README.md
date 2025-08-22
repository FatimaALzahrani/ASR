# Speech Recognition Training Pipeline

A modular and efficient pipeline for training speech recognition models with support for both general and personalized models.

## Features

- **Modular Design**: Each component is separated for easy maintenance
- **Audio Augmentation**: Random noise, volume, and time stretching
- **Feature Extraction**: MFCC + Mel Spectrogram features
- **Model Types**:
  - General model (works with all speakers)
  - Personalized models (speaker-specific)
- **Advanced Training**:
  - Label smoothing
  - Learning rate scheduling
  - Early stopping
  - Gradient clipping

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Ensure your data is organized as expected:

```
data/
├── processed/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   ├── mappings.json
│   └── speakers/
│       ├── أحمد/
│       ├── عاصم/
│       └── ...
└── enhanced/
    └── [enhanced audio files]
```

## Usage

### Basic Usage

```bash
python main.py
```

### Advanced Options

```bash
# Train only general model
python main.py --general-only --general-epochs 80

# Train only personalized models
python main.py --personalized-only --personalized-epochs 80

# Custom epochs for both
python main.py --general-epochs 80 --personalized-epochs 60
```

## Model Architecture

The `SpeechClassifier` uses a CNN-based architecture:

- 3 Convolutional blocks with BatchNorm and Dropout
- Global adaptive pooling
- 3 Fully connected layers for classification
- Input: 53 features (13 MFCC + 40 Mel spectrogram)

## File Descriptions

### `audio_processor.py`

Handles all audio-related operations:

- Loading audio files (with enhanced version preference)
- Resampling to target sample rate (16kHz)
- Audio augmentation for training
- Feature extraction (MFCC + Mel spectrogram)

### `dataset.py`

PyTorch Dataset implementation:

- Loads CSV files with audio paths and labels
- Applies augmentation during training
- Returns padded sequences for batch processing

### `model.py`

Neural network architecture:

- CNN-based feature extraction
- Fully connected classification layers
- Configurable dropout and input dimensions

### `evaluator.py`

Model evaluation utilities:

- Accuracy calculation
- Per-speaker performance analysis
- Detailed testing reports

### `trainer.py`

Training orchestration:

- Model initialization and setup
- Training loop with validation
- Best model saving and early stopping
- Learning rate scheduling

### `pipeline.py`

High-level training coordination:

- Manages training of all model types
- Results collection and comparison
- Summary reporting

### `main.py`

Entry point with command-line interface:

- Argument parsing
- Error handling
- Training execution

## Output

The pipeline will create:

- `models/` directory with saved model weights
- Console output showing training progress
- Final accuracy comparison between models
