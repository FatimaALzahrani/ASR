# Professional Arabic Speech Recognition System

A comprehensive Arabic speech recognition system with acoustic modeling, language modeling, and auto-correction capabilities.

## Features

- **Advanced Acoustic Model**: Multi-speaker support with comprehensive audio feature extraction
- **Language Model**: N-gram based Arabic language modeling for context-aware predictions
- **Auto-Correction**: Intelligent correction system for Arabic speech recognition errors
- **Ensemble Learning**: Combines multiple machine learning models for improved accuracy
- **Speaker-Specific Models**: Personalized models for individual speakers
- **Production Ready**: Modular design for easy deployment and maintenance

## Quick Start

### Training the System

```bash
python main.py --data_path /path/to/audio/data --output_dir models --min_samples 3
```

### Testing with Audio File

```bash
python main.py --data_path /path/to/audio/data --test_audio /path/to/test.wav --speaker أحمد
```

## Data Structure

Your audio data should be organized as follows:

```
data/
├── كلمة1/
│   ├── 1.wav
│   ├── 8.wav
│   └── 15.wav
├── كلمة2/
│   ├── 2.wav
│   ├── 9.wav
│   └── 16.wav
└── ...
```

## Speaker Mapping

The system supports the following speakers based on file numbering:

- **أحمد**: Files 0-6
- **عاصم**: Files 7-13
- **هيفاء**: Files 14-20
- **أسيل**: Files 21-28
- **وسام**: Files 29-36

## Usage Examples

### Programmatic Usage

```python
from asr_system import ProfessionalASRSystem

# Initialize system
asr = ProfessionalASRSystem()

# Train the system
results = asr.train(audio_data_path="data/", min_samples_per_word=3)

# Recognize speech
result = asr.recognize_speech("test_audio.wav", speaker="أحمد")
print(f"Predicted word: {result['final_prediction']}")
print(f"Confidence: {result['final_confidence']:.2f}")

# Save trained model
asr.save_system("trained_models/")

# Load existing model
asr.load_system("trained_models/")
```

### Individual Components

```python
# Use acoustic model only
from acoustic_model import AcousticModel

acoustic_model = AcousticModel()
acoustic_model.train_acoustic_models("data/")
prediction, confidence = acoustic_model.predict("test.wav")

# Use language model
from language_model import LanguageModel

lm = LanguageModel(n_gram=3)
lm.train(["نص عربي للتدريب", "نص آخر"])
probability = lm.get_word_probability("كلمة")

# Use auto-correction
from auto_correction import AutoCorrection

corrector = AutoCorrection(lm)
corrected, confidence = corrector.correct_word("كلمه")
```

## Audio Processing Features

The acoustic model extracts comprehensive features including:

- **Temporal Features**: RMS energy, amplitude statistics, duration
- **Spectral Features**: Centroid, rolloff, bandwidth, flatness, contrast
- **MFCC Features**: 20 coefficients with delta and delta-delta
- **Mel-scale Features**: 26-band mel-spectrogram
- **Harmonic Features**: Chroma, tonnetz, harmonic/percussive separation
- **Quality Features**: SNR estimation, clipping detection, silence ratio

## Model Performance

The system uses ensemble learning with multiple algorithms:

- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine
- Logistic Regression
- Extra Trees Classifier

Speaker-specific models are trained when sufficient data is available, falling back to global models otherwise.

## Configuration Options

### Acoustic Model Parameters

- `sample_rate`: Audio sampling rate (default: 22050 Hz)
- `duration`: Fixed audio duration for processing (default: 3.0 seconds)
- `min_samples_per_word`: Minimum samples required per word (default: 3)

### Language Model Parameters

- `n_gram`: N-gram order for language modeling (default: 3)

## Output Format

Recognition results include:

```python
{
    'acoustic_prediction': 'كلمة',
    'acoustic_confidence': 0.85,
    'final_prediction': 'كلمة',
    'final_confidence': 0.87,
    'corrections_applied': True,
    'correction_confidence': 0.90
}
```

## File Formats Supported

- WAV (.wav)
- MP3 (.mp3)
- M4A (.m4a)
- FLAC (.flac)

## Error Handling

The system includes comprehensive error handling:

- Audio file loading errors
- Feature extraction failures
- Model training exceptions
- Prediction errors with fallback responses

## Logging

The system provides detailed logging at multiple levels:

- INFO: Training progress and system status
- WARNING: Non-critical issues and fallbacks
- ERROR: Critical errors with context
- DEBUG: Detailed debugging information

## Performance Optimization

- Multi-threading support for model training
- Efficient feature caching
- Memory-optimized audio processing
- Scalable model architecture
