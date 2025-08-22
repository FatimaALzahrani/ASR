import os
import json
import torch
import pandas as pd
from trainer import ModelTrainer
from baseline_evaluator import BaselineEvaluator
from neural_models import HMM_DNN_Model, RNN_CNN_Model, EndToEnd_Model


class ComprehensiveTrainer(ModelTrainer):
    
    def __init__(self):
        super().__init__()
        self.baseline_evaluator = BaselineEvaluator(self)
    
    def run_comprehensive_training(self):
        print("Starting comprehensive fixed training...")
        print("=" * 70)
        
        baseline_acc = self.baseline_evaluator.run_baseline_comparison()
        
        results = {'Baseline k-NN': baseline_acc}
        
        os.makedirs('models', exist_ok=True)
        
        print("\n" + "="*50)
        print("Training HMM-DNN")
        print("="*50)
        
        train_loader, val_loader, test_loader = self.create_data_loaders(feature_type='mfcc', batch_size=32)
        hmm_model = HMM_DNN_Model(input_dim=39, hidden_dim=512, num_classes=len(self.label_encoder.classes_))
        hmm_accuracy = self.train_model(hmm_model, train_loader, val_loader, 'hmm_dnn', epochs=30)
        
        hmm_model.load_state_dict(torch.load('fixed_hmm_dnn_best.pth'))
        hmm_test_acc, hmm_speakers = self.detailed_evaluation(hmm_model, test_loader, 'HMM-DNN')
        results['HMM-DNN'] = hmm_test_acc
        
        print("\n" + "="*50)
        print("Training RNN-CNN")
        print("="*50)
        
        train_loader, val_loader, test_loader = self.create_data_loaders(feature_type='combined', batch_size=32)
        rnn_model = RNN_CNN_Model(input_dim=17, hidden_dim=256, num_classes=len(self.label_encoder.classes_))
        rnn_accuracy = self.train_model(rnn_model, train_loader, val_loader, 'rnn_cnn', epochs=35)
        
        rnn_model.load_state_dict(torch.load('fixed_rnn_cnn_best.pth'))
        rnn_test_acc, rnn_speakers = self.detailed_evaluation(rnn_model, test_loader, 'RNN-CNN')
        results['RNN-CNN'] = rnn_test_acc
        
        print("\n" + "="*50)
        print("Training End-to-End")
        print("="*50)
        
        train_loader, val_loader, test_loader = self.create_data_loaders(feature_type='mel_spectrogram', batch_size=24)
        e2e_model = EndToEnd_Model(input_dim=80, num_classes=len(self.label_encoder.classes_))
        e2e_accuracy = self.train_model(e2e_model, train_loader, val_loader, 'end_to_end', epochs=25)
        
        e2e_model.load_state_dict(torch.load('fixed_end_to_end_best.pth'))
        e2e_test_acc, e2e_speakers = self.detailed_evaluation(e2e_model, test_loader, 'End-to-End')
        results['End-to-End'] = e2e_test_acc
        
        print("\n" + "=" * 70)
        print("Final Results:")
        print("=" * 70)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, accuracy) in enumerate(sorted_results, 1):
            print(f"{i}. {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        best_model = max(results, key=results.get)
        best_accuracy = max(results.values())
        
        print(f"\nBest model: {best_model}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        
        final_results = {
            'model_accuracies': results,
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'dataset_info': {
                'train_samples': len(self.train_df),
                'val_samples': len(self.val_df),
                'test_samples': len(self.test_df),
                'num_words': len(self.label_encoder.classes_),
                'num_speakers': len(pd.concat([self.train_df, self.val_df, self.test_df])['speaker'].unique())
            }
        }
        
        with open('fixed_training_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\nGenerated files:")
        print(f"• fixed_hmm_dnn_best.pth - HMM-DNN model")
        print(f"• fixed_rnn_cnn_best.pth - RNN-CNN model")
        print(f"• fixed_end_to_end_best.pth - End-to-End model")
        print(f"• fixed_training_results.json - Comprehensive results")
        
        return results