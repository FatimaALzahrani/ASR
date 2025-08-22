# Fair Data Splitting Tool

A comprehensive tool for creating fair and balanced data splits that ensure equitable representation of all speakers across training, validation, and test sets. This tool addresses common bias issues in speech recognition datasets.

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your processed data exists in `data/processed/`:

   - `train.csv`
   - `validation.csv`
   - `test.csv`

2. Run the fair splitting tool:

```bash
python main.py
```

## Problem Addressed

### Unfair Distribution Issues

Many speech recognition datasets suffer from speaker imbalance, where:

- Some speakers dominate the test set
- Other speakers are underrepresented or missing
- This leads to biased evaluation and poor generalization

### Example of Unfair Distribution

```
Speaker A: 80% of test samples
Speaker B: 15% of test samples
Speaker C: 5% of test samples
Speaker D: 0% of test samples
Speaker E: 0% of test samples
```

## Solution: Fair Splitting Algorithm

### Per-Speaker Fair Distribution

The algorithm ensures each speaker is proportionally represented by:

1. **Analyzing Current Distribution**: Identifies imbalanced speaker representation
2. **Speaker-Level Splitting**: Splits each speaker's data individually
3. **Word-Level Balancing**: Ensures word variety per speaker
4. **Proportional Allocation**: Maintains consistent ratios across all speakers

### Splitting Strategy

For each speaker's words:

- **1 sample**: Goes to training (preserves rare data)
- **2 samples**: 1 training, 1 test
- **3 samples**: 2 training, 1 test
- **4+ samples**: Proportional split (70% train, 15% val, 15% test)

## Classes Overview

1. **DataAnalyzer**: Analyzes current data distribution and identifies imbalances
2. **FairSplitter**: Implements the fair splitting algorithm
3. **DataSaverFair**: Handles data export and creates backups
4. **DistributionComparer**: Compares old vs new distributions
5. **FairDataProcessor**: Orchestrates the entire fair splitting process

## Features

### Comprehensive Analysis

- **Distribution Assessment**: Detailed analysis of speaker representation
- **Fairness Metrics**: Coefficient of variation and min/max ratios
- **Word Coverage**: Ensures diverse vocabulary per speaker
- **Sample Preservation**: Maintains total sample count

### Smart Splitting

- **Per-Speaker Processing**: Individual handling of each speaker's data
- **Word-Level Granularity**: Splits at word level to maintain diversity
- **Minimum Sample Protection**: Ensures rare words aren't lost
- **Proportional Allocation**: Consistent ratios across speakers

### Data Safety

- **Automatic Backup**: Creates backup of original unfair splits
- **Reversible Process**: Can restore original data if needed
- **Statistics Tracking**: Detailed records of splitting decisions
- **Validation Checks**: Ensures data integrity throughout process

## Fairness Metrics

### Coefficient of Variation (CV)

- **< 0.3**: Fair distribution
- **0.3 - 0.5**: Moderately fair
- **> 0.5**: Unfair distribution

### Min/Max Ratio

- **Closer to 1.0**: More fair
- **Closer to 0.0**: Less fair

## Output Files

### Updated Data Splits

- `data/processed/train.csv` - Fair training set
- `data/processed/validation.csv` - Fair validation set
- `data/processed/test.csv` - Fair test set

### Backup and Statistics

- `data/processed/backup_unfair/` - Original unfair splits
- `data/processed/fair_split_stats.json` - Detailed splitting statistics

## Example Results

### Before Fair Splitting

```
Speaker Distribution in Test Set:
  Ahmed: 45 samples (65.2%)
  Asem: 15 samples (21.7%)
  Haifa: 8 samples (11.6%)
  Aseel: 1 sample (1.4%)
  Wessam: 0 samples (0.0%)
```

### After Fair Splitting

```
Speaker Distribution in Test Set:
  Ahmed: 18 samples (26.1%)
  Asem: 16 samples (23.2%)
  Haifa: 15 samples (21.7%)
  Aseel: 12 samples (17.4%)
  Wessam: 8 samples (11.6%)
```

## Benefits

### For Model Training

- **Reduced Bias**: Fair representation prevents speaker-specific overfitting
- **Better Generalization**: Models learn from all speakers equally
- **Reliable Evaluation**: Test results reflect true model performance
- **Consistent Metrics**: Evaluation metrics are stable across speakers

### For Research

- **Reproducible Results**: Consistent splitting methodology
- **Ethical AI**: Fair treatment of all demographic groups
- **Statistical Validity**: Proper experimental design
- **Publication Ready**: Meets standards for academic research

## Integration with Training Pipeline

After running fair splitting:

1. **Retrain Models**: Use the new fair splits for training
2. **Compare Results**: Evaluate improvement in fairness metrics
3. **Validate Performance**: Ensure maintained or improved accuracy
4. **Deploy Confidently**: Use models trained on fair data
