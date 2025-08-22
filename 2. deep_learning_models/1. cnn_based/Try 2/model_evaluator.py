import torch
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from speech_dataset import SpeechDataset, collate_fn
from speech_cnn_model import SpeechCNN
from torch.utils.data import DataLoader


class ModelEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open('data/processed/mappings.json', 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)
        
        self.num_classes = self.mappings['num_words']
        self.id_to_word = self.mappings['id_to_word']
    
    def load_model(self, model_path):
        """Load a trained model"""
        model = SpeechCNN(self.num_classes).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def evaluate_model(self, model, test_csv_path):
        dataset = SpeechDataset(test_csv_path, self.mappings)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        
        # Convert numeric labels to word labels
        target_words = [self.id_to_word[str(t)] for t in all_targets]
        pred_words = [self.id_to_word[str(p)] for p in all_predictions]
        
        # Classification report
        report = classification_report(target_words, pred_words, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(target_words, pred_words)
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'target_words': target_words,
            'pred_words': pred_words,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, target_words, title="Confusion Matrix"):
        plt.figure(figsize=(12, 10))
        
        # Get unique words for labels
        unique_words = sorted(list(set(target_words)))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_words, yticklabels=unique_words)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return plt.gcf()
    
    def evaluate_all_models(self):
        models_dir = Path('models')
        results = {}
        
        # Evaluate general model
        general_model_path = models_dir / 'general_best.pth'
        if general_model_path.exists():
            print("Evaluating general model...")
            model = self.load_model(general_model_path)
            results['general'] = self.evaluate_model(model, 'data/processed/test.csv')
            print(f"General model accuracy: {results['general']['accuracy']:.3f}")
        
        # Evaluate personalized models
        speakers = ['Ahmed', 'Asem', 'Haifa', 'Aseel', 'Wessam']
        
        for speaker in speakers:
            model_path = models_dir / f'personalized_{speaker}_best.pth'
            test_path = f'data/processed/speakers/{speaker}/test.csv'
            
            if model_path.exists() and Path(test_path).exists():
                print(f"Evaluating {speaker} personalized model...")
                model = self.load_model(model_path)
                results[f'personalized_{speaker}'] = self.evaluate_model(model, test_path)
                print(f"{speaker} model accuracy: {results[f'personalized_{speaker}']['accuracy']:.3f}")
        
        return results
    
    def generate_evaluation_report(self, results):
        """Generate comprehensive evaluation report"""
        report = {
            'model_performance': {},
            'best_performing_words': {},
            'worst_performing_words': {},
            'speaker_comparison': {}
        }
        
        for model_name, result in results.items():
            report['model_performance'][model_name] = {
                'accuracy': result['accuracy'],
                'macro_avg_f1': result['classification_report']['macro avg']['f1-score'],
                'weighted_avg_f1': result['classification_report']['weighted avg']['f1-score']
            }
        
        if 'general' in results:
            word_scores = {}
            for word, metrics in results['general']['classification_report'].items():
                if isinstance(metrics, dict) and 'f1-score' in metrics:
                    word_scores[word] = metrics['f1-score']
            
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            report['best_performing_words'] = dict(sorted_words[:10])
            report['worst_performing_words'] = dict(sorted_words[-10:])
        
        # Speaker comparison
        personalized_results = {k: v for k, v in results.items() if k.startswith('personalized_')}
        if personalized_results:
            speaker_accuracies = {}
            for model_name, result in personalized_results.items():
                speaker = model_name.replace('personalized_', '')
                speaker_accuracies[speaker] = result['accuracy']
            
            report['speaker_comparison'] = speaker_accuracies
        
        # Save report
        with open('results/evaluation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report


def main():
    print("Model Evaluation Pipeline")
    print("=" * 50)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    results = evaluator.evaluate_all_models()
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(results)
    
    # Plot confusion matrices
    for model_name, result in results.items():
        if len(result['target_words']) > 0:
            fig = evaluator.plot_confusion_matrix(
                result['confusion_matrix'], 
                result['target_words'],
                title=f"Confusion Matrix - {model_name}"
            )
            fig.savefig(f'results/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    print("\nEvaluation Summary:")
    print("=" * 30)
    for model_name, performance in report['model_performance'].items():
        print(f"{model_name}: {performance['accuracy']:.3f}")
    
    print(f"\nDetailed results saved in 'results/' directory")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()