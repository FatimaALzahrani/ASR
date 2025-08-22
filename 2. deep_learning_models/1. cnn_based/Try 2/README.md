# Speech Recognition Training Pipeline for Down Syndrome Children

A complete end-to-end pipeline for training speech recognition models specifically designed for children with Down syndrome. This system supports both general multi-speaker models and personalized single-speaker models.

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Ensure CUDA is available for GPU training (optional but recommended):

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Step 1: Data Processing

```bash
python main_data_processing.py
```

This will:

- Process audio files
- Filter words with minimum 3 samples
- Create both general and speaker-specific splits
- Generate mappings and save processed data

### Step 2: Model Training

```bash
python main_training.py
```

This will:

- Train a general multi-speaker model
- Train personalized models for each speaker
- Save best models in `models/` directory

## Features

### Data Processing

- **Minimum Sample Filtering**: Excludes words with fewer than specified samples
- **Speaker Identification**: Automatic speaker mapping from file numbers
- **Dual Split Strategy**: Creates both general and personalized data splits
- **Quality Metadata**: Incorporates speaker quality information

### Model Architecture

- **CNN-Based**: 1D Convolutional Neural Network optimized for MFCC features
- **MFCC Features**: 13-dimensional Mel-frequency cepstral coefficients
- **Adaptive Pooling**: Handles variable-length audio sequences
- **Dropout Regularization**: Prevents overfitting with 30% dropout

### Training Strategy

- **Dual Model Approach**:
  - General model: Trained on all speakers
  - Personalized models: One per speaker for better individual performance
- **Enhanced Audio Support**: Automatically uses enhanced audio if available
- **Learning Rate Scheduling**: Reduces learning rate on plateau
- **Best Model Saving**: Saves models with highest validation accuracy

## Model Performance

Based on the provided results:

### General Model

- **Test Accuracy**: 31.9%
- **Training**: Multi-speaker dataset
- **Use Case**: General speech recognition across all speakers

### Personalized Models

- **Ahmed**: 47.1% accuracy (weak speech quality)
- **Asem**: 58.3% accuracy (excellent speech quality)
- **Haifa**: 33.9% accuracy (good speech quality)
- **Aseel**: 48.2% accuracy (weak speech quality)
- **Wessam**: 62.5% accuracy (medium speech quality)

## Data Splits

### General Splits

- **Training**: 70% of data for words with 10+ samples
- **Validation**: 15% of data for model selection
- **Test**: 15% of data for final evaluation

### Personalized Splits

- **Training**: 80% of each speaker's data
- **Test**: 20% of each speaker's data
- **Single Sample Words**: Go entirely to training set

## File Organization

### Input Structure

```
data/
├── clean/
│   ├── word1/
│   │   ├── 001.wav
│   │   ├── 007.wav
│   │   └── ...
│   └── word2/
│       ├── 002.wav
│       └── ...
└── enhanced/  # Optional enhanced audio
    ├── word1/
    │   ├── enhanced_001.wav
    │   └── ...
    └── ...
```

### Output Structure

```
data/processed/
├── train.csv              # General training data
├── validation.csv         # General validation data
├── test.csv              # General test data
├── mappings.json         # Word and speaker mappings
└── speakers/
    ├── Ahmed/
    │   ├── train.csv
    │   └── test.csv
    ├── Asem/
    │   ├── train.csv
    │   └── test.csv
    └── ...

models/
├── general_best.pth           # Best general model
├── personalized_Ahmed_best.pth
├── personalized_Asem_best.pth
└── ...
```

## Classes Overview

1. **FinalDataProcessor**: Core data processing with speaker identification
2. **DataSplitterFinal**: Implements both general and speaker-specific splitting
3. **DataSaverFinal**: Handles data export and mapping creation
4. **SpeechDataset**: PyTorch dataset for audio loading and MFCC extraction
5. **SpeechCNN**: 1D CNN architecture for speech recognition
6. **SpeechTrainer**: Complete training pipeline with evaluation

## Model Architecture Details

### CNN Layers

```python
Input: (batch_size, sequence_length, 13)  # MFCC features
Conv1D(13 → 64) + ReLU + MaxPool1D(2)
Conv1D(64 → 128) + ReLU + MaxPool1D(2)
Conv1D(128 → 256) + ReLU + AdaptiveMaxPool1D(1)
Flatten → Linear(256 → 128) + ReLU + Dropout(0.3)
Linear(128 → num_classes)
```

### Training Configuration

- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 50
- **Scheduler**: ReduceLROnPlateau (patience=5)

## Speaker Quality Integration

The system incorporates speaker quality metadata:

- **Excellent**: Asem (58.3% accuracy)
- **Good**: Haifa (33.3% accuracy)
- **Medium**: Wessam (62.5% accuracy)
- **Weak**: Ahmed (47.1%), Aseel (48.2% accuracy)

Interestingly, some speakers with "weak" quality achieve good personalized model performance, suggesting the importance of personalization.

## Performance Optimization

### For Better Results:

1. **Data Augmentation**: Add noise, speed, and pitch variations
2. **Advanced Features**: Consider mel-spectrograms or raw waveforms
3. **Model Architecture**: Try RNNs, Transformers, or deeper CNNs
4. **Ensemble Methods**: Combine general and personalized models
5. **Transfer Learning**: Pre-train on larger speech datasets

### Memory and Speed:

- **GPU Training**: Significantly faster than CPU
- **Batch Processing**: Efficient data loading with PyTorch DataLoader
- **Model Checkpointing**: Save/load best models automatically
