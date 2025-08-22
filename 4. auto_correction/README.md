# Enhanced Auto Correction System

An advanced Arabic speech recognition auto-correction system designed to improve accuracy for children with special needs.

## Overview

This system applies multiple correction techniques to improve speech recognition accuracy:

- **Error Pattern Analysis**: Identifies and learns from common prediction errors
- **Phonetic Correction**: Uses Arabic phonetic similarity mapping
- **Similarity-based Correction**: Calculates text and phonetic similarity scores
- **Confidence-based Correction**: Corrects low-confidence predictions
- **Common Correction Rules**: Applies domain-specific correction rules

## Installation

### Prerequisites

```bash
pip install pandas numpy scikit-learn
```

### Required Files

1. Place your processed dataset at:

   ```
   C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Codes/Real Codes/01_data_processing/processed_dataset.csv
   ```

2. Place your trained model at:
   ```
   simplified_high_accuracy_model.pkl
   ```

## Usage

### Basic Usage

```python
from enhanced_auto_correction import EnhancedAutoCorrection
from model_utils import load_model_and_data

# Load data and model
df, model = load_model_and_data()

# Create auto corrector
auto_corrector = EnhancedAutoCorrection()

# Build language model
vocabulary = list(set(df['word']))
auto_corrector.build_language_model(vocabulary)

# Apply corrections
corrected_predictions, stats = auto_corrector.apply_auto_correction(
    predictions, vocabulary, confidence_scores
)
```

### Running the Complete System

```bash
python main.py
```

## Features

### EnhancedAutoCorrection Class

**Key Methods:**

- `analyze_error_patterns()`: Analyzes prediction errors and builds error patterns database
- `build_language_model()`: Creates word frequency model with common Arabic words for children
- `calculate_similarity_score()`: Computes similarity between words using text, phonetic, and length features
- `get_correction_candidates()`: Finds potential corrections using multiple strategies
- `apply_auto_correction()`: Applies corrections to predictions with detailed statistics

**Phonetic Support:**

- Comprehensive Arabic character phonetic similarity mapping
- Handles common phonetic confusions in Arabic speech
- Supports 33 Arabic characters with similarity relationships

**Correction Strategies:**

1. **Pattern-based**: Uses learned error patterns from training data
2. **Similarity-based**: Finds similar words using multiple similarity metrics
3. **Common corrections**: Applies predefined correction rules for children's vocabulary
4. **Confidence-based**: Corrects predictions below confidence threshold

### Model Utilities

- **Automatic model loading**: Handles different model storage formats
- **Fallback model creation**: Creates simple model if loading fails
- **Data preprocessing**: Prepares features for model inference

### Data Simulation

- **Error simulation**: Creates realistic prediction errors for testing
- **Phonetic errors**: Simulates common Arabic phonetic mistakes
- **Configurable error rates**: Adjustable error simulation parameters

### Report Generation

- **Comprehensive reporting**: Detailed analysis of correction performance
- **Performance metrics**: Accuracy improvement, error analysis, correction statistics
- **Markdown output**: Professional formatted reports

## Output Files

1. **final_auto_correction_results.json**: Numerical results and statistics
2. **final_auto_correction_comprehensive_report.md**: Detailed analysis report

## Configuration

### Adjustable Parameters

```python
auto_corrector.confidence_threshold = 0.6    # Confidence threshold for correction
auto_corrector.similarity_threshold = 0.6    # Minimum similarity for candidates
```

### Error Simulation

```python
error_rate = 0.25  # 25% error rate for simulation
```

## Performance

**Expected Improvements:**

- Accuracy improvement: 4-8%
- Error reduction: 15-25%
- Phonetic error correction: 60-80%

**System Requirements:**

- Memory: ~100MB for vocabulary and models
- Processing: Real-time correction capability
- Compatibility: Works with any speech recognition model

## Customization

### Adding New Phonetic Rules

```python
auto_corrector.phonetic_similarity['ض'] = ['ظ', 'ص']
```

### Adding Common Corrections

```python
auto_corrector.common_corrections['كتاب'] = ['كتب', 'كتابة']
```

### Adjusting Similarity Weights

```python
# In calculate_similarity_score method
total_score = (text_similarity * 0.4 +
              phonetic_similarity * 0.4 +
              length_similarity * 0.2)
```
