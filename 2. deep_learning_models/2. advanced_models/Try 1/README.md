# Fixed Training System for Down Syndrome Speech Recognition

A comprehensive training pipeline for Down Syndrome children speech recognition using multiple neural network architectures.

## Features

### Smart Audio File Search

- Automatically searches for audio files in multiple locations
- Handles different naming conventions
- Robust error handling for missing files

### Multiple Model Architectures

- **HMM-DNN**: Hybrid model with attention mechanism
- **RNN-CNN**: Combined recurrent and convolutional architecture
- **End-to-End**: Transformer-based model with global attention

### Comprehensive Evaluation

- Baseline k-NN comparison
- Per-speaker performance analysis
- Detailed accuracy metrics
- Model comparison and ranking

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python main.py
```

### Expected Data Structure

```
data/
├── processed/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
└── clean/
    └── [word_folders]/
        └── [audio_files.wav]
```

## Model Training Process

1. **Data Loading**: Loads preprocessed CSV files
2. **Baseline Evaluation**: k-NN classifier for comparison
3. **HMM-DNN Training**: MFCC features with attention
4. **RNN-CNN Training**: Combined features with bidirectional processing
5. **End-to-End Training**: Mel spectrogram with transformer layers

## Output Files

- `fixed_hmm_dnn_best.pth` - Best HMM-DNN model
- `fixed_rnn_cnn_best.pth` - Best RNN-CNN model
- `fixed_end_to_end_best.pth` - Best End-to-End model
- `fixed_training_results.json` - Complete results and metrics

## Model Architecture Details

### HMM-DNN Model

- Input: MFCC features (39 dimensions)
- Architecture: DNN → BiLSTM → Attention → Classification
- Features: Batch normalization, dropout, gradient clipping

### RNN-CNN Model

- Input: Combined features (17 dimensions)
- Architecture: CNN → BiGRU → Attention → Classification
- Features: Conv1D layers, bidirectional processing

### End-to-End Model

- Input: Mel spectrogram (80 dimensions)
- Architecture: CNN Encoder → Transformer → Global Attention → Classification
- Features: Multi-head attention, GELU activation

## Training Configuration

- **Batch Sizes**: 24-32 depending on model complexity
- **Learning Rate**: 0.001 with StepLR scheduling
- **Regularization**: Dropout, weight decay, gradient clipping
- **Early Stopping**: Patience of 10 epochs
- **Data Augmentation**: Light audio augmentation for training

## Performance Metrics

- Overall accuracy per model
- Per-speaker breakdown
- Training/validation curves
- Best model identification
- Comprehensive result logging
