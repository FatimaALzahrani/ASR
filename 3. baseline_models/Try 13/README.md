# Advanced Down Syndrome Children Speech Recognition System

A comprehensive automatic speech recognition system specifically designed for children with Down syndrome, utilizing the complete database with advanced machine learning techniques.

## Project Overview

This system provides:

- **Complete Database Utilization**: 101 words, 1307 recordings, 5 speakers
- **Advanced Feature Extraction**: 700+ sophisticated audio features
- **Intelligent Ensemble Models**: Adaptive machine learning algorithms
- **Personalized Processing**: Speaker-specific optimizations
- **Comprehensive Analysis**: Detailed performance evaluation

## Installation

### Requirements

```bash
pip install numpy pandas librosa scikit-learn xgboost lightgbm scipy matplotlib seaborn
```

### Dependencies

- **Python 3.7+**
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Librosa**: Audio processing
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting
- **LightGBM**: Efficient gradient boosting
- **SciPy**: Scientific computing
- **Matplotlib/Seaborn**: Visualization

## Usage

### Basic Usage

```python
from asr_system import AdvancedASRSystem

# Initialize system
asr = AdvancedASRSystem()

# Run comprehensive analysis
results = asr.run_comprehensive_analysis()
```

### Custom Configuration

```python
# Custom data and output paths
asr = AdvancedASRSystem(
    data_path="/path/to/audio/data",
    output_path="/path/to/results"
)

results = asr.run_comprehensive_analysis()
```

### Configuration

Modify `config.py` to adjust:

- **Data paths**
- **Speaker profiles**
- **Word categories**
- **Difficulty levels**
- **Quality mappings**

## Features

### Audio Processing

- **Adaptive preprocessing** based on speaker characteristics
- **Noise reduction** and signal enhancement
- **Age-specific filtering** for children vs adults
- **Quality-based optimization** for unclear speech

### Feature Extraction

- **MFCC features** with multiple configurations
- **Spectral features** (centroid, rolloff, bandwidth, etc.)
- **F0 features** with age-adaptive ranges
- **Temporal features** (rhythm, tempo, silence analysis)
- **Harmonic features** (chroma, harmonic-percussive separation)
- **Down syndrome specific features**
- **Statistical and complexity features**

### Machine Learning

- **Multiple algorithms**: Random Forest, XGBoost, LightGBM, SVM, etc.
- **Intelligent ensemble**: Voting classifiers with optimization
- **Feature selection**: Automated optimal feature selection
- **Cross-validation**: Robust model evaluation
- **Outlier removal**: Data quality enhancement

### Data Augmentation

- **Intelligent augmentation** based on speaker quality
- **Multiple techniques**: Gaussian noise, scaling, shifting, rotation
- **Adaptive methods** for different speech difficulties
- **Quality-aware processing**

## Output

The system generates:

- **JSON results**: Detailed performance metrics
- **Pickle models**: Trained models and preprocessors
- **Text reports**: Comprehensive analysis reports
- **Performance analysis**: Error analysis and insights

## Results Structure

```json
{
  "speaker_name": {
    "accuracy": 0.85,
    "f1_score": 0.83,
    "precision": 0.84,
    "recall": 0.82,
    "best_model": "VotingWeighted",
    "samples": 250,
    "words": 45
  }
}
```

## Scientific Contributions

### Methodological Innovations

- First comprehensive system using complete available database
- Adaptive methodology with individual child characteristics
- Advanced word classification by difficulty and category
- Specialized intelligent data augmentation techniques
- Competitive results with comprehensive error analysis

### Technical Advances

- **700+ advanced features** with intelligent selection
- **Speaker-adaptive processing** (age, IQ, speech clarity)
- **Quality-aware augmentation** strategies
- **Advanced ensemble methods** with weighted voting
- **Comprehensive evaluation** metrics

## Performance Benchmarks

- **Target Accuracy**: >60% for competitive performance
- **Excellent Performance**: >80% (world-leading)
- **Very Good Performance**: 70-79% (publication-ready)
- **Good Performance**: 60-69% (significant contribution)

## File Descriptions

### Core Components

- **`main.py`**: Entry point for system execution
- **`asr_system.py`**: Main coordinator class
- **`config.py`**: Centralized configuration management

### Data Processing

- **`data_loader.py`**: Handles data loading and preprocessing
- **`audio_processor.py`**: Audio signal processing and enhancement
- **`feature_extractor.py`**: Comprehensive feature extraction
- **`data_augmentation.py`**: Intelligent data augmentation

### Machine Learning

- **`model_trainer.py`**: Advanced model training and ensemble creation
- **`results_analyzer.py`**: Results analysis and scientific reporting
- **`utils.py`**: Supporting utility functions
