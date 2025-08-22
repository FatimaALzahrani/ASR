# Maximum Data Usage Training System

A comprehensive system for maximizing data utilization in small speech recognition datasets using multiple training strategies.

## Training Strategies

### 1. Simple 90/10 Split

- Uses 90% of combined data for training
- 10% for testing
- Maximizes training data while maintaining evaluation capability

### 2. 5-Fold Cross Validation

- Divides data into 5 folds
- Trains 5 different models
- Provides robust performance estimation with confidence intervals

### 3. Leave-One-Speaker-Out (LOSO)

- Trains on 4 speakers, tests on 1
- Evaluates speaker-independent performance
- Critical for real-world deployment assessment

### 4. Full Data with Strong Regularization

- Uses all available data for training
- Applies heavy regularization to prevent overfitting
- Maximum data utilization approach

## Features

### Data Utilization

- Combines train, validation, and test sets
- Maximizes available samples for training
- Provides detailed data distribution analysis

### Training Optimization

- Early stopping to prevent overfitting
- Gradient clipping for stable training
- Learning rate scheduling
- Multiple regularization techniques

### Comprehensive Evaluation

- Per-strategy performance comparison
- Statistical significance testing
- Speaker-specific analysis
- Best strategy recommendation

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Ensure processed data is available:

```
data/processed/
├── train.csv
├── validation.csv
└── test.csv
```

## Usage

### Basic Usage

```bash
python main.py
```

### Expected Output

The system will:

1. Load and combine all available data
2. Run Strategy 1: Simple 90/10 split
3. Run Strategy 2: 5-fold cross validation
4. Run Strategy 3: Leave-one-speaker-out
5. Run Strategy 4: Full data with regularization
6. Compare all strategies and recommend best approach

### Output Files

- `max_data_usage_results.json` - Detailed results for all strategies
- `full_data_model.pth` - Model trained on full dataset (Strategy 4)

## Results Interpretation

### Strategy Comparison Metrics

- **Simple 90/10**: Single accuracy score
- **Cross Validation**: Mean ± standard deviation
- **LOSO**: Per-speaker accuracy breakdown
- **Full Data**: Training accuracy with regularization

### Best Strategy Selection

The system automatically identifies the best performing strategy based on:

- Highest accuracy for simple splits
- Mean accuracy for cross-validation
- Average accuracy across speakers for LOSO
- Final training accuracy for full data approach
