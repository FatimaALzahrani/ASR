# Enhanced Speech Recognition Models Pipeline

A comprehensive PyTorch-based pipeline for advanced speech recognition using state-of-the-art deep learning models including HMM-DNN, RNN-CNN, and End-to-End architectures.

## Overview

This project implements three enhanced deep learning models for speech recognition tasks, each optimized for different aspects of audio processing and feature extraction. The pipeline includes advanced feature engineering, multiple model architectures, and comprehensive evaluation metrics.

## Model Architectures

### 1. Enhanced HMM-DNN Model (`hmm_dnn_model.py`)

Hybrid model combining Hidden Markov Models with Deep Neural Networks.

**Architecture Components:**

- **DNN Feature Processor**: Multi-layer neural network for frame-level processing
- **LSTM Sequence Modeling**: Bidirectional LSTM for temporal dependencies
- **Attention Mechanism**: Weighted attention over sequence outputs
- **Advanced Regularization**: Batch normalization and dropout layers

**Key Features:**

- MFCC + Delta + Delta-Delta features (39-dimensional)
- Frame-level processing with sequence-level classification
- Attention-based temporal aggregation
- Designed for phoneme-level accuracy

### 2. Enhanced RNN-CNN Model (`rnn_cnn_model.py`)

Combined Recurrent and Convolutional architecture for feature learning.

**Architecture Components:**

- **CNN Feature Extractor**: 1D convolutions for local pattern detection
- **RNN Sequence Processor**: Multi-layer GRU for sequential modeling
- **Advanced Attention**: Multi-layer attention with tanh activation
- **Hierarchical Classification**: Multi-stage feature processing

**Key Features:**

- Combined spectral and temporal features (17-dimensional)
- Local and global feature extraction
- Bidirectional sequence processing
- Robust to speaker variations

### 3. Enhanced End-to-End Model (`end_to_end_model.py`)

Transformer-based architecture for direct audio-to-text mapping.

**Architecture Components:**

- **Multi-layer CNN Encoder**: Progressive feature abstraction
- **Transformer Encoder**: Self-attention mechanisms for long-range dependencies
- **Global Attention Pooling**: Weighted combination of all sequence positions
- **Deep Classification Head**: Multi-layer classifier with regularization

**Key Features:**

- Mel-spectrogram input (80-dimensional)
- Self-attention for parallel processing
- End-to-end differentiable training
- State-of-the-art accuracy potential

## Feature Types

### MFCC Features

- **Mel-Frequency Cepstral Coefficients**: 13 coefficients
- **Delta Coefficients**: First-order derivatives
- **Delta-Delta Coefficients**: Second-order derivatives
- **Total Dimensions**: 39 features per frame

### Combined Features

- **MFCC**: 13 coefficients
- **Spectral Centroid**: Frequency center of mass
- **Spectral Bandwidth**: Frequency spread
- **Spectral Rolloff**: High-frequency content measure
- **Zero Crossing Rate**: Temporal characteristics
- **Total Dimensions**: 17 features per frame

### Mel-Spectrogram Features

- **Mel-Scale Filter Banks**: 80 mel-frequency bins
- **Log Power**: Logarithmic compression
- **Temporal Resolution**: High-resolution time-frequency representation
- **Total Dimensions**: 80 features per frame

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- PyTorch >= 1.9.0
- librosa >= 0.8.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Usage

### Quick Start

```bash
python main.py
```

### Custom Configuration

```python
from enhanced_models_trainer import EnhancedModelsTrainer

# Initialize trainer with custom path
trainer = EnhancedModelsTrainer('path/to/your/dataset.csv')

# Run comprehensive comparison
results = trainer.run_comprehensive_comparison()
```

### Training Individual Models

```python
from enhanced_models_trainer import EnhancedModelsTrainer
from hmm_dnn_model import ImprovedHMM_DNN_Model

# Initialize trainer
trainer = EnhancedModelsTrainer()

# Create data loaders
train_loader, test_loader = trainer.create_data_loaders(feature_type='mfcc')

# Create and train model
model = ImprovedHMM_DNN_Model(input_dim=39, hidden_dim=512, num_classes=72)
accuracy = trainer.train_model(model, train_loader, test_loader, 'hmm_dnn')
```

## Training Process

### Data Preprocessing

1. **Audio Loading**: Load audio files using librosa
2. **Feature Extraction**: Extract specified feature type
3. **Sequence Normalization**: Pad/truncate to fixed length
4. **Data Augmentation**: Apply light augmentation for training data

### Training Configuration

- **Loss Function**: CrossEntropyLoss with label smoothing (0.1)
- **Optimizer**: AdamW with weight decay (1e-4)
- **Scheduler**: CosineAnnealingLR for learning rate scheduling
- **Regularization**: Gradient clipping (max_norm=1.0)
- **Early Stopping**: Patience of 8 epochs

### Model-Specific Settings

#### HMM-DNN Model

- **Epochs**: 30
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Feature Type**: MFCC (39-dim)

#### RNN-CNN Model

- **Epochs**: 35
- **Learning Rate**: 0.0008
- **Batch Size**: 32
- **Feature Type**: Combined (17-dim)

#### End-to-End Model

- **Epochs**: 40
- **Learning Rate**: 0.0005
- **Batch Size**: 16
- **Feature Type**: Mel-spectrogram (80-dim)

## Output Files

The training pipeline generates comprehensive outputs:

1. **Model Checkpoints**:

   - `best_hmm_dnn_enhanced.pth`
   - `best_rnn_cnn_enhanced.pth`
   - `best_end_to_end_enhanced.pth`

2. **Results and Analysis**:

   - `enhanced_models_comparison.json`
   - `*_training_curves.png`

3. **Performance Metrics**:
   - Overall accuracy
   - Per-speaker analysis
   - Classification reports
   - Training curves

### Results Format

```json
{
  "model_accuracies": {
    "HMM-DNN Enhanced": 0.41,
    "RNN-CNN Enhanced": 0.44,
    "End-to-End Enhanced": 0.02
  },
  "best_model": "RNN-CNN Enhanced",
  "best_accuracy": 0.44,
  "detailed_results": {
    "HMM-DNN Enhanced": {
      "accuracy": 0.41,
      "speaker_analysis": {...},
      "classification_report": {...}
    }
  }
}
```

## Advanced Features

### Data Augmentation

- **Noise Injection**: Gaussian noise with adaptive variance
- **Time Stretching**: Speed variations (±5%)
- **Pitch Shifting**: Frequency modifications (±2 semitones)
- **Conditional Application**: Smart augmentation based on data quality

### Attention Mechanisms

- **Temporal Attention**: Focus on important time frames
- **Feature Attention**: Emphasize relevant feature dimensions
- **Multi-head Attention**: Parallel attention computations
- **Global Attention**: Long-range dependency modeling

### Regularization Techniques

- **Dropout**: Various rates across different layers
- **Batch Normalization**: Stable training dynamics
- **Weight Decay**: L2 regularization on parameters
- **Label Smoothing**: Improved generalization
- **Gradient Clipping**: Stable gradient flow

## Performance Analysis

### Evaluation Metrics

- **Overall Accuracy**: Global classification performance
- **Per-Speaker Analysis**: Individual speaker performance
- **Per-Word Analysis**: Word-level recognition rates
- **Confusion Matrices**: Detailed error analysis
- **Training Curves**: Loss and accuracy progression

### Benchmarking

- **Baseline Comparison**: k-NN baseline (27.58% accuracy)
- **Model Ranking**: Performance-based ordering
- **Improvement Metrics**: Quantified enhancements
- **Statistical Significance**: Confidence intervals

## Customization

### Adding New Features

```python
# In enhanced_audio_dataset.py
def extract_custom_features(self, audio, sr):
    # Your custom feature extraction
    custom_features = your_feature_function(audio, sr)
    return custom_features.T
```

### Modifying Model Architecture

```python
# Example: Adding more layers to HMM-DNN
class CustomHMM_DNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # Add your custom layers
        self.extra_layer = nn.Linear(hidden_dim, hidden_dim)
```

### Adjusting Training Parameters

```python
# In enhanced_models_trainer.py
def train_model(self, model, train_loader, test_loader, model_name,
                epochs=50, lr=0.0005):  # Custom parameters
    # Modified training configuration
```

## Hardware Requirements

### Minimum Requirements

- **CPU**: Multi-core processor (4+ cores)
- **RAM**: 16GB minimum
- **Storage**: 5GB free space
- **GPU**: Optional but recommended

### Recommended Configuration

- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: Version 11.0 or higher
- **RAM**: 32GB for large datasets
- **Storage**: SSD for faster I/O

### Performance Expectations

- **CPU Training**: 2-4 hours per model
- **GPU Training**: 30-60 minutes per model
- **Memory Usage**: 4-8GB during training
- **Inference Speed**: <100ms per sample

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```python
# Reduce batch size
train_loader = DataLoader(dataset, batch_size=8)  # Reduce from 16/32
```

#### Poor Convergence

1. Adjust learning rate
2. Modify batch size
3. Check data quality
4. Verify feature scaling

#### Feature Extraction Errors

- Ensure librosa installation
- Check audio file formats
- Verify sampling rates
- Handle missing files

### Performance Optimization

#### Memory Optimization

```python
# Enable memory optimization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

#### Training Acceleration

1. Use mixed precision training
2. Increase batch size
3. Optimize data loading
4. Use compiled models

## Model Deployment

### Saving Models

```python
# Complete model saving
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'accuracy': accuracy
}, 'complete_model.pth')
```

### Loading for Inference

```python
# Load trained model
model = ImprovedHMM_DNN_Model(input_dim=39, hidden_dim=512, num_classes=72)
model.load_state_dict(torch.load('best_hmm_dnn_enhanced.pth'))
model.eval()

# Inference
with torch.no_grad():
    output = model(features)
    prediction = torch.argmax(output, dim=1)
```

### Production Pipeline

```python
def predict_speech(audio_path, model, feature_extractor, label_encoder):
    # Load and process audio
    audio, sr = librosa.load(audio_path, sr=16000)
    features = feature_extractor.extract_features(audio, sr)
    features_tensor = torch.FloatTensor(features).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(features_tensor)
        prediction = torch.argmax(output, dim=1)
        word = label_encoder.inverse_transform(prediction.cpu().numpy())

    return word[0]
```

## Research Applications

### Academic Use

- Speech recognition research
- Deep learning methodology studies
- Feature engineering experiments
- Architectural comparisons

### Industrial Applications

- Voice assistant development
- Speech-to-text systems
- Language learning tools
- Accessibility technologies
