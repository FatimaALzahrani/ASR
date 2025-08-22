import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_basic_metrics(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted')
    }
    return metrics


def calculate_per_class_metrics(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(y_true)
    
    f1_scores = f1_score(y_true, y_pred, labels=labels, average=None)
    precision_scores = precision_score(y_true, y_pred, labels=labels, average=None)
    recall_scores = recall_score(y_true, y_pred, labels=labels, average=None)
    
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[str(label)] = {
            'f1_score': f1_scores[i],
            'precision': precision_scores[i],
            'recall': recall_scores[i]
        }
    
    return per_class_metrics


def generate_confusion_matrix(y_true, y_pred, labels=None, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", save_path=None):
    cm = generate_confusion_matrix(y_true, y_pred, labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def evaluate_model_performance(model, X_test, y_test, labels=None):
    y_pred = model.predict(X_test)
    
    basic_metrics = calculate_basic_metrics(y_test, y_pred)
    per_class_metrics = calculate_per_class_metrics(y_test, y_pred, labels)
    
    evaluation_results = {
        'basic_metrics': basic_metrics,
        'per_class_metrics': per_class_metrics,
        'predictions': y_pred.tolist(),
        'actual': y_test.tolist()
    }
    
    return evaluation_results


def compare_models(models_results):
    comparison_df = pd.DataFrame()
    
    for model_name, results in models_results.items():
        if 'basic_metrics' in results:
            metrics = results['basic_metrics']
            comparison_df[model_name] = [
                metrics['accuracy'],
                metrics['f1_score'],
                metrics['precision'],
                metrics['recall']
            ]
    
    comparison_df.index = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    return comparison_df


def calculate_confidence_metrics(y_true, y_pred, confidence_scores, threshold=0.5):
    high_confidence_mask = confidence_scores >= threshold
    
    if high_confidence_mask.sum() == 0:
        return {
            'high_confidence_accuracy': 0.0,
            'high_confidence_samples': 0,
            'coverage': 0.0
        }
    
    high_conf_true = y_true[high_confidence_mask]
    high_conf_pred = y_pred[high_confidence_mask]
    
    high_confidence_accuracy = accuracy_score(high_conf_true, high_conf_pred)
    high_confidence_samples = high_confidence_mask.sum()
    coverage = high_confidence_samples / len(y_true)
    
    return {
        'high_confidence_accuracy': high_confidence_accuracy,
        'high_confidence_samples': int(high_confidence_samples),
        'coverage': coverage,
        'threshold': threshold
    }


def evaluate_speaker_specific_performance(y_true, y_pred, speakers):
    speaker_performance = {}
    
    unique_speakers = np.unique(speakers)
    
    for speaker in unique_speakers:
        speaker_mask = speakers == speaker
        
        if speaker_mask.sum() > 0:
            speaker_true = y_true[speaker_mask]
            speaker_pred = y_pred[speaker_mask]
            
            speaker_metrics = calculate_basic_metrics(speaker_true, speaker_pred)
            speaker_performance[speaker] = {
                **speaker_metrics,
                'n_samples': int(speaker_mask.sum())
            }
    
    return speaker_performance


def calculate_word_level_accuracy(y_true, y_pred, words):
    word_performance = {}
    
    unique_words = np.unique(words)
    
    for word in unique_words:
        word_mask = words == word
        
        if word_mask.sum() > 0:
            word_true = y_true[word_mask]
            word_pred = y_pred[word_mask]
            
            word_accuracy = accuracy_score(word_true, word_pred)
            word_performance[word] = {
                'accuracy': word_accuracy,
                'n_samples': int(word_mask.sum())
            }
    
    return word_performance


def generate_detailed_report(evaluation_results, save_path=None):
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("DETAILED EVALUATION REPORT")
    report_lines.append("=" * 60)
    
    basic_metrics = evaluation_results.get('basic_metrics', {})
    report_lines.append("\nBASIC METRICS:")
    report_lines.append("-" * 30)
    for metric, value in basic_metrics.items():
        report_lines.append(f"{metric.capitalize()}: {value:.4f}")
    
    per_class_metrics = evaluation_results.get('per_class_metrics', {})
    if per_class_metrics:
        report_lines.append("\nPER-CLASS METRICS:")
        report_lines.append("-" * 30)
        for class_name, metrics in per_class_metrics.items():
            report_lines.append(f"\nClass: {class_name}")
            for metric, value in metrics.items():
                report_lines.append(f"  {metric}: {value:.4f}")
    
    confidence_metrics = evaluation_results.get('confidence_metrics', {})
    if confidence_metrics:
        report_lines.append("\nCONFIDENCE METRICS:")
        report_lines.append("-" * 30)
        for metric, value in confidence_metrics.items():
            if isinstance(value, float):
                report_lines.append(f"{metric}: {value:.4f}")
            else:
                report_lines.append(f"{metric}: {value}")
    
    speaker_performance = evaluation_results.get('speaker_performance', {})
    if speaker_performance:
        report_lines.append("\nSPEAKER PERFORMANCE:")
        report_lines.append("-" * 30)
        for speaker, metrics in speaker_performance.items():
            report_lines.append(f"\nSpeaker: {speaker}")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric}: {value}")
    
    report_text = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Report saved to: {save_path}")
    else:
        print(report_text)
    
    return report_text