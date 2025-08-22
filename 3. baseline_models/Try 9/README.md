# Speech Recognition System for Children with Down Syndrome

A comprehensive speech recognition system specifically designed for children with Down syndrome, implementing advanced acoustic models, language models, and automatic correction systems.

## Features

- **Advanced Acoustic Model**: Ensemble-based acoustic recognition optimized for Down syndrome speech patterns
- **Language Model**: Context-aware language modeling with speaker-specific adaptations
- **Automatic Corrector**: Phonetic pattern-based error correction system
- **Integrated System**: Complete pipeline combining all models for real-time speech recognition
- **Speaker Adaptation**: Personalized models for individual speakers
- **Multi-Model Training**: Support for traditional ML and advanced boosting algorithms

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with CLI

```bash
# Check system status
python main.py status

# Train all models
python main.py train

# Train specific models
python main.py train --models acoustic language

# Recognize single audio file
python main.py recognize --audio-file path/to/audio.wav --speaker "أحمد"

# Recognize directory of files
python main.py recognize --audio-dir path/to/directory --speaker "هيفاء"

# Test system
python main.py test
```

### Programmatic Usage

#### Training Models

```python
from train_models import ModelTrainer

trainer = ModelTrainer()
trainer.run_complete_training()

# Train individual models
trainer.train_individual_model('acoustic')
trainer.train_individual_model('language')
trainer.train_individual_model('corrector')
```

#### Using Integrated System

```python
from integrated_system import IntegratedSpeechSystem

system = IntegratedSpeechSystem()
system.load_models()

# Single file recognition
result = system.recognize_speech("audio_file.wav", speaker="أحمد")

# Sequence recognition
results = system.recognize_speech_sequence(
    ["file1.wav", "file2.wav"],
    speaker="هيفاء"
)
```

#### Individual Components

```python
from models.acoustic_model import AdvancedAcousticModel
from models.language_model import LanguageModel
from models.corrector_model import AutoCorrector
from models.ensemble_model import EnsembleModel

# Individual model training
acoustic_model = AdvancedAcousticModel()
acoustic_model.train()

language_model = LanguageModel()
language_model.train()

corrector = AutoCorrector()
corrector.train()

ensemble_model = EnsembleModel()
ensemble_model.train()
```

## Components

### Core Components

- **BaseModel**: Abstract base class for all models
- **FeatureExtractor**: Advanced audio feature extraction
- **DataProcessor**: Data preprocessing and preparation

### Models

- **AcousticModel**: Ensemble acoustic recognition model
- **LanguageModel**: N-gram language model with context awareness
- **CorrectorModel**: Phonetic pattern-based error correction
- **EnsembleModel**: Combined model integration

### Utilities

- **AudioUtils**: Audio processing utilities
- **Evaluation**: Model evaluation metrics
- **FileUtils**: File handling utilities

## Configuration

Edit `config/settings.py` to customize:

- Feature extraction parameters
- Model hyperparameters
- File paths and directories
- Training configurations

## Data Format

The system expects audio files with corresponding transcriptions:

```
processed_data/
├── speaker1/
│   ├── word1.wav
│   ├── word2.wav
│   └── ...
└── speaker2/
    ├── word1.wav
    └── ...
```

## Refactoring and Architecture

This project has been refactored from the original monolithic files into a clean, modular architecture:

### Key Improvements

- **Separation of Concerns**: Each class has a single responsibility
- **Modular Design**: Components can be used independently or together
- **Clean Interfaces**: Consistent API across all models
- **Enhanced Maintainability**: Easy to modify, test, and extend
- **Improved CLI**: Comprehensive command-line interface
- **Better Error Handling**: Robust error handling throughout
- **English Messaging**: All print statements converted to English

### Original Files Restructured

- `advanced_acoustic_model.py` → `models/acoustic_model.py`
- `language_model_and_corrector.py` → `models/language_model.py` + `models/corrector_model.py`
- `speech_recognition_models.py` → `models/ensemble_model.py`
- `integrated_speech_system.py` → `integrated_system.py` (cleaned and enhanced)

### Architecture Benefits

- **Scalability**: Easy to add new models or features
- **Testing**: Each component can be tested independently
- **Reusability**: Components can be reused in other projects
- **Configuration**: Centralized configuration management
- **Utilities**: Shared utilities for common operations

## Performance

The system achieves optimized recognition accuracy for Down syndrome speech patterns through:

- Phonetic pattern analysis specific to Down syndrome
- Speaker-specific adaptations and personalization
- Context-aware corrections using n-gram models
- Ensemble model combinations for improved accuracy
- Advanced feature extraction (MFCC, spectral, prosodic features)

## Research Basis

Built on scientific research for Down syndrome speech recognition, incorporating:

- Acoustic pattern analysis specific to Down syndrome speech characteristics
- Language model adaptations for common speech difficulties
- Error correction based on phonetic substitution patterns
- Speaker-specific model customization and adaptation
- Evidence-based feature selection for this population

## System Components

### Core Components

- **BaseModel**: Abstract base class providing consistent interface
- **FeatureExtractor**: Extracts 80+ audio features including MFCC, spectral, and prosodic
- **DataProcessor**: Handles data loading, preprocessing, and feature selection

### Models

- **AcousticModel**: Ensemble-based acoustic recognition (Random Forest + XGBoost + LightGBM)
- **LanguageModel**: N-gram language model with context awareness and speaker adaptation
- **CorrectorModel**: Phonetic pattern-based automatic error correction
- **EnsembleModel**: Multi-algorithm ensemble for maximum accuracy

### Utilities

- **AudioUtils**: Audio loading, preprocessing, and validation
- **Evaluation**: Comprehensive evaluation metrics and reporting
- **FileUtils**: File operations, organization, and management
