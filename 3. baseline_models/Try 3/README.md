# Improved Machine Learning Pipeline with Strong Regularization

A comprehensive machine learning pipeline for audio classification with enhanced regularization techniques and modular architecture.

## File Descriptions

### Core Files

#### `main.py`

Entry point for the application. Simply imports and runs the main trainer.

```python
from improved_models_trainer import ImprovedModelsTrainer

def main():
    trainer = ImprovedModelsTrainer()
    trainer.run_complete_pipeline()
```

#### `improved_models_trainer.py`

Main orchestrator that coordinates all pipeline components.

**Key Features:**

- Manages overall pipeline execution
- Coordinates data flow between components
- Handles initialization and cleanup

#### `data_loader.py`

Handles data loading and speaker-based splitting.

**Key Features:**

- CSV file loading with error handling
- Speaker-based train/test splitting to prevent data leakage
- Maintains speaker metadata

#### `feature_extractor.py`

Manages feature extraction and preprocessing.

**Key Features:**

- Extracts audio features (duration, RMS energy, SNR, quality)
- Extracts text features (word length, character counts)
- Applies robust scaling and label encoding
- Filters common words between train/test sets

#### `model_factory.py`

Creates and configures machine learning models.

**Key Features:**

- Creates 8 regularized base models
- Configures ensemble voting classifiers
- Applies strong regularization parameters

#### `model_trainer.py`

Handles model training and evaluation.

**Key Features:**

- Stratified cross-validation
- Comprehensive performance metrics
- Overfitting detection and reporting

#### `hyperparameter_tuner.py`

Optimizes model hyperparameters using grid search.

**Key Features:**

- Grid search for Random Forest and SVM
- Cross-validation during tuning
- Returns optimized models

#### `results_analyzer.py`

Creates visualizations and performance analysis.

**Key Features:**

- 6 comprehensive visualization plots
- Model comparison charts
- Performance summary displays

#### `results_saver.py`

Manages result persistence and summaries.

**Key Features:**

- Saves results in JSON format
- Creates performance summaries
- Identifies best performing models

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

### Quick Start

```bash
python main.py
```

### Custom Data Path

```python
from improved_models_trainer import ImprovedModelsTrainer

trainer = ImprovedModelsTrainer('path/to/your/dataset.csv')
trainer.run_complete_pipeline()
```

### Using Individual Components

```python
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from model_factory import ModelFactory
from model_trainer import ModelTrainer

# Load data
loader = DataLoader('dataset.csv')
loader.load_data()
train_data, test_data = loader.split_by_speakers()

# Extract features
extractor = FeatureExtractor()
X_train, X_test, y_train, y_test, words = extractor.preprocess_data(train_data, test_data)

# Create and train models
factory = ModelFactory()
models = factory.create_regularized_models()

trainer = ModelTrainer(X_train, X_test, y_train, y_test)
results = trainer.train_and_evaluate_models(models)
```

## Models Included

### Base Models (8 models)

- **Random Forest (Regularized)**: Limited depth and samples
- **SVM (High Regularization)**: Strong C regularization
- **Logistic Regression (L2)**: L2 penalty with low C
- **Ridge Classifier**: Ridge regularization
- **Gradient Boosting (Regularized)**: Low learning rate and limited estimators
- **KNN (Optimized)**: Distance-weighted neighbors
- **Naive Bayes**: Probabilistic classifier
- **Extra Trees (Regularized)**: Extremely randomized trees

### Ensemble Models (2 models)

- **Voting Classifier (Hard)**: Hard voting ensemble
- **Voting Classifier (Soft)**: Soft voting with probabilities

### Tuned Models (2 models)

- **Random Forest (Tuned)**: Grid search optimized
- **SVM (Tuned)**: Grid search optimized

## Data Requirements

Input CSV must contain:

- `speaker`: Speaker identifier
- `word`: Target word/label
- `duration`: Audio duration
- `rms_energy`: RMS energy value
- `snr`: Signal-to-noise ratio
- `quality_score`: Audio quality metric

## Output Files

1. **improved_models_comprehensive_analysis.png**: Visualization plots
2. **improved_models_comprehensive_results.json**: Detailed results

## Configuration

### Modifying Model Parameters

Edit `model_factory.py` to adjust regularization:

```python
'Random Forest (Regularized)': RandomForestClassifier(
    n_estimators=30,        # Reduce for more regularization
    max_depth=3,           # Reduce for more regularization
    min_samples_split=20,  # Increase for more regularization
    # ... other parameters
)
```

### Changing Cross-Validation

Edit `model_trainer.py`:

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Change n_splits
```

### Adjusting Hyperparameter Tuning

Edit `hyperparameter_tuner.py`:

```python
rf_param_grid = {
    'n_estimators': [10, 20, 30, 50],  # Add more values
    'max_depth': [2, 3, 4, 5],        # Extend range
    # ... other parameters
}
```

## Extending the Pipeline

### Adding New Models

1. Edit `model_factory.py`:

```python
def create_regularized_models(self):
    models = {
        # ... existing models
        'Your New Model': YourModelClass(
            param1=value1,
            param2=value2
        )
    }
    return models
```

### Adding New Features

1. Edit `feature_extractor.py` in the `extract_features` method:

```python
def extract_features(self, data):
    # ... existing features
    features.extend([
        your_new_feature1,
        your_new_feature2
    ])
```

### Adding New Visualizations

1. Edit `results_analyzer.py` in the `create_comprehensive_plots` method:

```python
def create_comprehensive_plots(self, all_results):
    # ... existing plots
    axes[row, col].your_new_plot()
```

## Performance Monitoring

The system automatically tracks:

- **Cross-validation scores** with standard deviation
- **Overfitting gaps** (train - test accuracy)
- **F1-scores** for balanced evaluation
- **Training time** and resource usage

### Overfitting Detection

- **High**: Gap > 0.2 (⚠️ High overfitting)
- **Moderate**: Gap 0.1-0.2 (⚠️ Moderate overfitting)
- **Low**: Gap < 0.1 (✅ Low overfitting)

## Best Practices

1. **Always use speaker-based splits** for audio data
2. **Apply strong regularization** to prevent overfitting
3. **Use stratified cross-validation** for reliable estimates
4. **Monitor overfitting gaps** during model selection
5. **Combine multiple models** using ensemble methods

## Troubleshooting

### Import Errors

Ensure all files are in the same directory:

```bash
ls -la *.py
```

### Memory Issues

Reduce model complexity in `model_factory.py`:

- Decrease `n_estimators`
- Reduce `max_depth`
- Increase `min_samples_split`

### Poor Performance

1. Check data quality in the CSV file
2. Verify feature extraction logic
3. Adjust regularization parameters
4. Try different cross-validation strategies

## Development Guidelines

When modifying the codebase:

1. **Maintain class separation**: Each class should have a single responsibility
2. **Handle errors gracefully**: Use try-catch blocks with meaningful error messages
3. **Update documentation**: Modify this README when adding new features
4. **Test individually**: Each component can be tested independently
5. **Follow naming conventions**: Use descriptive variable and method names
