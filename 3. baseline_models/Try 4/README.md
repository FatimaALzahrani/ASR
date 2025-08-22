# Rapid Audio Classification Training Pipeline

A fast and efficient PyTorch-based audio classification system with personalized speaker models and advanced training techniques.

## Overview

This project implements a rapid training pipeline for audio classification tasks using deep learning. The system features both basic and personalized models, with the personalized model incorporating speaker-specific embeddings for improved performance.

## Architecture

### Core Components

#### `audio_dataset.py`

Custom PyTorch Dataset for audio files with augmentation support.

**Features:**

- Loads audio files using librosa
- Normalizes audio length to fixed size
- Applies data augmentation (noise, speed, pitch)
- Handles error cases gracefully

**Key Methods:**

- `__getitem__()`: Returns audio tensor, label, and speaker
- `normalize_length()`: Pads or crops audio to fixed length
- `apply_augmentation()`: Applies random audio transformations

#### `simple_audio_classifier.py`

1D CNN model for basic audio classification.

**Architecture:**

- 3-layer 1D convolutional feature extractor
- Batch normalization and dropout for regularization
- Adaptive pooling for variable input sizes
- Dense classifier head

**Features:**

- Handles raw audio waveforms
- Efficient 1D convolutions for temporal patterns
- Dropout for overfitting prevention

#### `personalized_model.py`

Enhanced model with speaker-specific embeddings.

**Features:**

- Extends base model with speaker embeddings
- Combines audio features with speaker information
- Improved performance for speaker-dependent tasks

**Architecture:**

- Base audio feature extractor
- Speaker embedding layer
- Combined feature fusion
- Enhanced classification head

#### `rapid_trainer.py`

Main training orchestrator with comprehensive evaluation.

**Capabilities:**

- Data loading and preprocessing
- Model training with optimization
- Detailed performance evaluation
- Result saving and analysis

## Models

### Basic Audio Classifier

- **Input**: Raw audio waveform (16kHz, 1 second)
- **Architecture**: 1D CNN with 3 convolutional layers
- **Output**: Word classification probabilities
- **Training**: 8 epochs with Adam optimizer

### Personalized Model

- **Input**: Audio + speaker ID
- **Architecture**: Basic model + speaker embeddings
- **Features**: Speaker-aware classification
- **Training**: 12 epochs with reduced learning rate

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

## Usage

### Quick Start

```bash
python main.py
```

### Custom Configuration

```python
from rapid_trainer import RapidTrainer

# Initialize trainer with custom data path
trainer = RapidTrainer('path/to/your/dataset.csv')

# Run training pipeline
results = trainer.run_rapid_training()
```

### Using Individual Components

```python
from audio_dataset import AudioDataset
from simple_audio_classifier import SimpleAudioClassifier
from personalized_model import PersonalizedModel

# Create dataset
dataset = AudioDataset(dataframe, label_encoder, augment=True)

# Create basic model
model = SimpleAudioClassifier(num_classes=10)

# Create personalized model
personalized = PersonalizedModel(model)
```

## Data Requirements

The input CSV file must contain:

- `file_path`: Path to audio file
- `word`: Target word/label
- `speaker`: Speaker identifier

### Supported Audio Formats

- WAV, MP3, FLAC (via librosa)
- 16kHz sampling rate (auto-converted)
- Variable length (auto-normalized to 1 second)

## Training Process

### Phase 1: Basic Model Training

1. Load and preprocess data
2. Create data loaders with augmentation
3. Train 1D CNN classifier
4. Evaluate performance
5. Save best model checkpoint

### Phase 2: Personalized Model Training

1. Load pre-trained basic model
2. Add speaker embedding layers
3. Fine-tune with speaker information
4. Evaluate enhanced performance
5. Save improved model

### Data Augmentation

- **Noise Injection**: Gaussian noise (σ=0.005)
- **Speed Variation**: 0.9x to 1.1x speed changes
- **Pitch Shifting**: ±2 semitone variations

## Output Files

The pipeline generates several output files:

1. **best_basic_model.pth**: Basic model weights
2. **best_personalized_model.pth**: Personalized model weights
3. **detailed_results_basic.json**: Basic model evaluation
4. **detailed_results_personalized.json**: Personalized model evaluation
5. **rapid_training_summary.json**: Complete training summary

### Results Structure

```json
{
  "training_completed": true,
  "basic_model": {
    "accuracy": 0.85,
    "detailed_results": {
      "overall_accuracy": 0.85,
      "speaker_analysis": {...},
      "word_analysis": {...}
    }
  },
  "personalized_model": {
    "accuracy": 0.92,
    "detailed_results": {...}
  },
  "improvement": 0.07,
  "improvement_percentage": 8.2
}
```

## Performance Monitoring

### Metrics Tracked

- **Overall Accuracy**: Global classification performance
- **Speaker-wise Accuracy**: Performance per individual speaker
- **Word-wise Accuracy**: Performance per target word
- **Training Loss**: Convergence monitoring
- **Learning Rate**: Optimization tracking

### Evaluation Process

1. Stratified train/test split (80/20)
2. Cross-entropy loss optimization
3. Accuracy-based model selection
4. Detailed per-speaker analysis
5. Comprehensive performance reporting

## Customization

### Modifying Model Architecture

Edit `simple_audio_classifier.py`:

```python
class SimpleAudioClassifier(nn.Module):
    def __init__(self, num_classes, input_size=16000):
        # Modify layers here
        self.feature_extractor = nn.Sequential(
            # Add/remove layers
            nn.Conv1d(1, 128, kernel_size=160, stride=32),  # Larger filters
            # ... additional layers
        )
```

### Adjusting Training Parameters

Edit `rapid_trainer.py`:

```python
def train_basic_model(self, epochs=15, lr=0.0005):  # Modify defaults
    # Training configuration
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

### Custom Data Augmentation

Edit `audio_dataset.py`:

```python
def apply_augmentation(self, audio):
    # Add custom augmentations
    if np.random.random() < 0.4:
        # Your custom augmentation
        audio = your_augmentation_function(audio)
    return audio
```

## Hardware Requirements

### Minimum Requirements

- CPU: Multi-core processor
- RAM: 8GB minimum
- Storage: 2GB free space

### Recommended for GPU Acceleration

- GPU: NVIDIA GPU with CUDA support
- VRAM: 4GB minimum
- CUDA: Version 11.0 or higher

### Performance Expectations

- **CPU Training**: ~30 minutes per model
- **GPU Training**: ~5-10 minutes per model
- **Memory Usage**: ~2-4GB during training

## Advanced Features

### Automatic Device Detection

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### Learning Rate Scheduling

- StepLR with decay every 5/7 epochs
- Gamma factor of 0.5 for gradual reduction

### Model Checkpointing

- Automatic saving of best models
- Loss-based model selection
- Resume training capability

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```python
# Reduce batch size
train_loader = DataLoader(dataset, batch_size=16)  # Reduce from 32
```

#### Audio Loading Errors

- Ensure audio files exist and are readable
- Check supported formats (WAV, MP3, FLAC)
- Verify file permissions

#### Poor Performance

1. Increase training epochs
2. Adjust learning rate
3. Add more data augmentation
4. Check data quality and labels

### Performance Optimization

#### Memory Optimization

```python
# Clear cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

#### Faster Training

1. Use GPU acceleration
2. Increase batch size (if memory allows)
3. Reduce audio length for faster processing
4. Use mixed precision training

## Model Deployment

### Saving Models

```python
# Save complete model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = SimpleAudioClassifier(num_classes=10)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Inference Pipeline

```python
def predict_audio(model, audio_path, label_encoder):
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = normalize_length(audio, 16000)
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)

    with torch.no_grad():
        output = model(audio_tensor)
        prediction = torch.argmax(output, dim=1)
        word = label_encoder.inverse_transform(prediction.cpu().numpy())

    return word[0]
```

## Contributing

### Development Guidelines

1. Follow modular architecture
2. Add comprehensive error handling
3. Include detailed documentation
4. Test with various audio formats
5. Maintain backward compatibility

### Adding New Models

1. Inherit from `nn.Module`
2. Implement `forward()` method
3. Add to trainer class
4. Update documentation

## Best Practices

### Data Preparation

1. Ensure consistent audio quality
2. Balance speaker representation
3. Verify label accuracy
4. Use stratified splitting

### Training Strategy

1. Start with basic model
2. Fine-tune personalized version
3. Monitor overfitting
4. Use appropriate regularization

### Evaluation

1. Use held-out test set
2. Analyze per-speaker performance
3. Check confusion matrices
4. Validate on unseen speakers
