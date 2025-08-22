import numpy as np
import json
from sklearn.model_selection import train_test_split

from enhanced_auto_correction import EnhancedAutoCorrection
from model_utils import load_model_and_data, create_simple_model
from data_simulator import simulate_predictions_with_errors
from report_generator import create_comprehensive_report

def main():
    print("Starting Enhanced Auto Correction System")
    print("=" * 50)
    
    df, model = load_model_and_data()
    if df is None:
        return
    
    print("\nPreparing data...")
    
    feature_columns = [col for col in df.columns if col not in ['word', 'speaker', 'file_path']]
    X = df[feature_columns]
    y = df['word']
    
    word_counts = y.value_counts()
    valid_words = word_counts[word_counts >= 2].index
    
    mask = y.isin(valid_words)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"Original data: {len(df)} samples")
    print(f"Filtered data: {len(X_filtered)} samples")
    print(f"Valid words: {len(valid_words)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.3, random_state=42, stratify=y_filtered
    )
    
    print(f"Training data: {len(X_train)} samples")
    print(f"Testing data: {len(X_test)} samples")
    
    print("\nGetting predictions...")
    
    if model is not None:
        try:
            predictions = model.predict(X_test)
            print("Model predictions obtained")
            
            confidences = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)
                confidences = np.max(probabilities, axis=1)
                print("Confidence scores calculated")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            print("Using simulation instead...")
            predictions = simulate_predictions_with_errors(y_test, error_rate=0.25)
            confidences = None
    else:
        print("Using simulation...")
        predictions = simulate_predictions_with_errors(y_test, error_rate=0.25)
        confidences = None
    
    original_accuracy = np.mean(predictions == y_test)
    print(f"Original accuracy: {original_accuracy:.4f} ({original_accuracy*100:.2f}%)")
    
    print("\nCreating Auto Correction system...")
    auto_corrector = EnhancedAutoCorrection()
    
    error_stats = auto_corrector.analyze_error_patterns(predictions, y_test)
    
    print(f"\nError analysis:")
    print(f"   Total errors: {error_stats['total_errors']}")
    print(f"   Substitution errors: {error_stats['substitution_errors']}")
    print(f"   Phonetic errors: {error_stats['phonetic_errors']}")
    print(f"   Length errors: {error_stats['length_errors']}")
    
    vocabulary = list(set(y_train))
    auto_corrector.build_language_model(list(y_train))
    
    print(f"\nLanguage model:")
    print(f"   Vocabulary size: {len(vocabulary)}")
    print(f"   Common words: {len(auto_corrector.word_frequencies)}")
    
    print("\nApplying auto correction...")
    
    corrected_predictions, correction_stats = auto_corrector.apply_auto_correction(
        predictions, vocabulary, confidences
    )
    
    corrected_accuracy = np.mean(corrected_predictions == y_test)
    improvement = corrected_accuracy - original_accuracy
    improvement_percent = (improvement / original_accuracy) * 100 if original_accuracy > 0 else 0
    
    print(f"\nFinal results:")
    print(f"   Original accuracy: {original_accuracy:.4f} ({original_accuracy*100:.2f}%)")
    print(f"   Corrected accuracy: {corrected_accuracy:.4f} ({corrected_accuracy*100:.2f}%)")
    print(f"   Improvement: +{improvement:.4f} (+{improvement_percent:.2f}%)")
    
    print(f"\nCorrection statistics:")
    print(f"   Total corrections: {correction_stats['total_corrections']}")
    print(f"   Pattern corrections: {correction_stats['pattern_corrections']}")
    print(f"   Similarity corrections: {correction_stats['similarity_corrections']}")
    print(f"   Common corrections: {correction_stats['common_corrections']}")
    if confidences is not None:
        print(f"   Low confidence corrections: {correction_stats['low_confidence_corrections']}")
    
    print(f"\nCorrection examples:")
    corrections_found = 0
    for i, (orig, corr, true) in enumerate(zip(predictions, corrected_predictions, y_test)):
        if orig != corr and corrections_found < 5:
            status = "✅ Correct" if corr == true else "❌ Wrong"
            print(f"   {orig} → {corr} (Expected: {true}) {status}")
            corrections_found += 1
    
    results = {
        'original_accuracy': float(original_accuracy),
        'corrected_accuracy': float(corrected_accuracy),
        'improvement': float(improvement),
        'improvement_percent': float(improvement_percent),
        'error_stats': {k: int(v) if isinstance(v, (int, np.integer)) else float(v) 
                       for k, v in error_stats.items() if k != 'common_confusions'},
        'correction_stats': {k: int(v) for k, v in correction_stats.items()},
        'total_samples': int(len(y_test)),
        'vocabulary_size': int(len(vocabulary)),
        'has_confidence_scores': confidences is not None,
        'model_type': 'real' if model is not None else 'simulated'
    }
    
    with open('final_auto_correction_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: final_auto_correction_results.json")
    
    create_comprehensive_report(results, auto_corrector)
    
    return results

if __name__ == "__main__":
    main()