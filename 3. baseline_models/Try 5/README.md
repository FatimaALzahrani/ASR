# Audio Quality Enhancement and Training System

A comprehensive system for enhancing audio quality and training machine learning models on audio data.

## Project Architecture

This project follows a modular architecture with each class in its own file for maximum maintainability.

### Key Benefits of This Architecture:

- **Separation of Concerns**: Each class has a single responsibility
- **Easy Testing**: Individual components can be tested in isolation
- **Modular Development**: Components can be developed and maintained independently
- **Flexible Usage**: Import only the components you need
- **Scalability**: Easy to add new features or modify existing ones

## Project Structure

## Project Structure

### Audio Enhancement Classes

#### `speech_boundary_detector.py`

- **SpeechBoundaryDetector**: Detects speech boundaries in audio recordings

#### `noise_reducer.py`

- **NoiseReducer**: Removes noise from audio using spectral analysis

#### `volume_normalizer.py`

- **VolumeNormalizer**: Normalizes audio volume levels

#### `audio_quality_enhancer.py`

- **AudioQualityEnhancer**: Main class that orchestrates the enhancement process

#### `dataset_enhancer.py`

- **DatasetEnhancer**: Handles enhancement of entire audio datasets

### Machine Learning Classes

#### `speaker_identifier.py`

- **SpeakerIdentifier**: Identifies speakers from filename patterns

#### `feature_extractor.py`

- **FeatureExtractor**: Extracts audio features (MFCC, spectral features, etc.)

#### `dataset_loader.py`

- **DatasetLoader**: Loads and preprocesses audio datasets

#### `model_trainer.py`

- **ModelTrainer**: Trains multiple ML models (RandomForest, GradientBoosting, SVM)

#### `results_comparator.py`

- **ResultsComparator**: Compares model performance before/after enhancement

#### `enhanced_training_pipeline.py`

- **EnhancedTrainingPipeline**: Main pipeline that orchestrates the training process

### Configuration and Utilities

#### `config.py`

Configuration classes:

- **AudioConfig**: Audio processing parameters
- **FeatureConfig**: Feature extraction settings
- **TrainingConfig**: ML training parameters
- **SpeakerConfig**: Speaker identification mappings
- **PathConfig**: Default file paths

#### `utils.py`

Utility classes:

- **FileUtils**: File and directory operations
- **DataUtils**: Data processing utilities
- **ReportUtils**: Report generation and formatting
- **ValidationUtils**: Input validation functions
- **LoggerUtils**: Logging and messaging

#### `main.py`

Main entry point with command-line interface

## Key Features

### Audio Enhancement

- Smart audio trimming to remove silence
- Noise reduction using spectral filtering
- Volume normalization
- Batch processing of entire datasets
- Quality-aware processing based on analysis

### Machine Learning

- Multiple model training and comparison
- Feature extraction from audio signals
- Speaker-specific model training
- Performance comparison analysis
- Stratified data splitting

## Dependencies

```
numpy
librosa
soundfile
pandas
scipy
scikit-learn
pathlib
joblib
```

## Usage

### Command Line Interface

#### Audio Enhancement Only

```bash
python main.py --mode enhance --input-dir "path/to/input" --output-dir "path/to/output"
```

#### Model Training Only

```bash
python main.py --mode train --output-dir "path/to/enhanced/data"
```

#### Full Pipeline

```bash
python main.py --mode full --input-dir "path/to/input" --output-dir "path/to/output"
```

### Programmatic Usage

#### Audio Enhancement

```python
from dataset_enhancer import DatasetEnhancer

enhancer = DatasetEnhancer()
results = enhancer.enhance_dataset(
    input_dir="path/to/input",
    output_dir="path/to/output"
)
```

#### Model Training

```python
from enhanced_training_pipeline import EnhancedTrainingPipeline

pipeline = EnhancedTrainingPipeline()
results = pipeline.run_training_comparison()
```

#### Individual Components

```python
from speech_boundary_detector import SpeechBoundaryDetector
from noise_reducer import NoiseReducer
from volume_normalizer import VolumeNormalizer
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer

# Use individual components
detector = SpeechBoundaryDetector()
noise_reducer = NoiseReducer()
volume_normalizer = VolumeNormalizer()
```

#### Custom Configuration

```python
from config import AudioConfig, TrainingConfig

# Modify configurations
AudioConfig.TARGET_SAMPLE_RATE = 16000
TrainingConfig.TEST_SIZE = 0.3
```

### Examples

See `examples.py` for comprehensive usage examples of individual components and full workflows.

## Input Data Structure

Expected directory structure:

```
data/
├── word1/
│   ├── 0.wav
│   ├── 1.wav
│   └── ...
├── word2/
│   ├── 0.wav
│   ├── 1.wav
│   └── ...
└── ...
```

## Speaker Mapping

The system identifies speakers based on filename numbers:

- أحمد: 0-6
- عاصم: 7-13
- هيفاء: 14-20
- أسيل: 21-28
- وسام: 29-36

## Output

### Enhancement Output

- Enhanced audio files in organized directory structure
- Processing reports in CSV format
- Enhancement statistics

### Training Output

- Model comparison results
- Performance metrics
- JSON report with detailed analysis

## Configuration

### Audio Enhancement Parameters

- Target sample rate: 22050 Hz
- Noise reduction factor: 0.5
- Target RMS: 0.02
- Silence margin: 0.05 seconds

### Training Parameters

- Test split: 20%
- Feature selection: 80 best features
- Cross-validation: Stratified when possible
- Random state: 42 for reproducibility

## Model Types

Three different models are trained and compared:

1. **Random Forest**: Ensemble method with 100 estimators
2. **Gradient Boosting**: Boosting method with 100 estimators
3. **SVM**: Support Vector Machine with RBF kernel

## Performance Metrics

The system tracks:

- Accuracy scores
- Training/testing split information
- Feature importance
- Processing time
- Enhancement statistics (duration reduction, RMS improvement)

## Error Handling

- Robust file processing with error reporting
- Graceful handling of missing files
- Quality-based processing decisions
- Comprehensive logging of all operations
