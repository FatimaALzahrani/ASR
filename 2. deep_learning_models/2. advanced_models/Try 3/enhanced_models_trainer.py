import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from enhanced_audio_dataset import EnhancedAudioDataset
from hmm_dnn_model import ImprovedHMM_DNN_Model
from rnn_cnn_model import ImprovedRNN_CNN_Model
from end_to_end_model import ImprovedEndToEndModel


class EnhancedModelsTrainer:
    def __init__(self, data_path="processed_dataset.csv"):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        print(f"Using device: {self.device}")
        self.load_data()
        self.setup_encoders()
        
    def load_data(self):
        print("Loading processed data...")
        
        self.df = pd.read_csv(self.data_path)
        
        word_counts = self.df['word'].value_counts()
        common_words = word_counts[word_counts >= 8]
        
        self.filtered_df = self.df[self.df['word'].isin(common_words.index)]
        
        print(f"Loaded {len(self.filtered_df)} samples from {len(common_words)} words")
        print(f"Average samples per word: {len(self.filtered_df) / len(common_words):.1f}")
        
    def setup_encoders(self):
        self.word_encoder = LabelEncoder()
        self.word_encoder.fit(self.filtered_df['word'])
        
        self.speaker_encoder = LabelEncoder()
        self.speaker_encoder.fit(self.filtered_df['speaker'])
        
        print(f"Word encoder setup for {len(self.word_encoder.classes_)} words")
        print(f"Speaker encoder setup for {len(self.speaker_encoder.classes_)} speakers")
        
    def create_data_loaders(self, feature_type='mfcc', test_size=0.25, batch_size=32):
        print(f"Creating data loaders for features: {feature_type}")
        
        try:
            train_df, test_df = train_test_split(
                self.filtered_df, test_size=test_size, random_state=42, 
                stratify=self.filtered_df['word']
            )
        except ValueError:
            train_df, test_df = train_test_split(
                self.filtered_df, test_size=test_size, random_state=42
            )
        
        train_dataset = EnhancedAudioDataset(train_df, self.word_encoder, 
                                           feature_type=feature_type, augment=True)
        test_dataset = EnhancedAudioDataset(test_df, self.word_encoder, 
                                          feature_type=feature_type, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=0, pin_memory=True)
        
        print(f"Training data: {len(train_dataset)} samples")
        print(f"Testing data: {len(test_dataset)} samples")
        
        return train_loader, test_loader
    
    def train_model(self, model, train_loader, test_loader, model_name, epochs=25, lr=0.001):
        print(f"Training model {model_name}...")
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_accuracy = 0
        patience_counter = 0
        max_patience = 8
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (features, labels, speakers) in enumerate(train_loader):
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            test_accuracy = self.evaluate_model(model, test_loader)
            train_accuracy = train_correct / train_total
            avg_loss = total_loss / len(train_loader)
            
            train_losses.append(avg_loss)
            test_accuracies.append(test_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_accuracy:.4f}, Test Acc={test_accuracy:.4f}, Loss={avg_loss:.4f}")
            
            scheduler.step()
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), f'best_{model_name}_enhanced.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Best accuracy for {model_name}: {best_accuracy:.4f}")
        
        self.save_training_curves(train_losses, test_accuracies, model_name)
        
        return best_accuracy
    
    def evaluate_model(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels, speakers in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def detailed_evaluation(self, model, test_loader, model_name):
        model.eval()
        all_predictions = []
        all_labels = []
        all_speakers = []
        
        with torch.no_grad():
            for features, labels, speakers in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_speakers.extend(speakers)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        
        class_names = self.word_encoder.classes_
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, output_dict=True)
        
        speaker_analysis = self.analyze_by_speaker(all_predictions, all_labels, all_speakers)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'speaker_analysis': speaker_analysis
        }
    
    def analyze_by_speaker(self, predictions, labels, speakers):
        speaker_results = {}
        
        for speaker in set(speakers):
            speaker_mask = [s == speaker for s in speakers]
            speaker_preds = [predictions[i] for i, mask in enumerate(speaker_mask) if mask]
            speaker_labels = [labels[i] for i, mask in enumerate(speaker_mask) if mask]
            
            if speaker_preds:
                accuracy = accuracy_score(speaker_labels, speaker_preds)
                speaker_results[speaker] = {
                    'accuracy': accuracy,
                    'total_samples': len(speaker_preds)
                }
        
        return speaker_results
    
    def save_training_curves(self, train_losses, test_accuracies, model_name):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title(f'{model_name} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(test_accuracies)
        plt.title(f'{model_name} - Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_comparison(self):
        print("Starting comprehensive comparison of enhanced models...")
        print("=" * 80)
        
        results = {}
        detailed_results = {}
        
        print("\n" + "="*50)
        print("Training Enhanced HMM-DNN Model")
        print("="*50)
        
        train_loader, test_loader = self.create_data_loaders(feature_type='mfcc', batch_size=32)
        hmm_dnn_model = ImprovedHMM_DNN_Model(input_dim=39, hidden_dim=512, 
                                            num_classes=len(self.word_encoder.classes_))
        hmm_dnn_accuracy = self.train_model(hmm_dnn_model, train_loader, test_loader, 
                                          'hmm_dnn', epochs=30, lr=0.001)
        results['HMM-DNN Enhanced'] = hmm_dnn_accuracy
        
        hmm_dnn_model.load_state_dict(torch.load('best_hmm_dnn_enhanced.pth'))
        detailed_results['HMM-DNN Enhanced'] = self.detailed_evaluation(hmm_dnn_model, test_loader, 'hmm_dnn')
        
        print("\n" + "="*50)
        print("Training Enhanced RNN-CNN Model")
        print("="*50)
        
        train_loader, test_loader = self.create_data_loaders(feature_type='combined', batch_size=32)
        rnn_cnn_model = ImprovedRNN_CNN_Model(input_dim=17, hidden_dim=256, 
                                            num_classes=len(self.word_encoder.classes_))
        rnn_cnn_accuracy = self.train_model(rnn_cnn_model, train_loader, test_loader, 
                                          'rnn_cnn', epochs=35, lr=0.0008)
        results['RNN-CNN Enhanced'] = rnn_cnn_accuracy
        
        rnn_cnn_model.load_state_dict(torch.load('best_rnn_cnn_enhanced.pth'))
        detailed_results['RNN-CNN Enhanced'] = self.detailed_evaluation(rnn_cnn_model, test_loader, 'rnn_cnn')
        
        print("\n" + "="*50)
        print("Training Enhanced End-to-End Model")
        print("="*50)
        
        train_loader, test_loader = self.create_data_loaders(feature_type='mel_spectrogram', batch_size=16)
        e2e_model = ImprovedEndToEndModel(input_dim=80, num_classes=len(self.word_encoder.classes_))
        e2e_accuracy = self.train_model(e2e_model, train_loader, test_loader, 
                                      'end_to_end', epochs=40, lr=0.0005)
        results['End-to-End Enhanced'] = e2e_accuracy
        
        e2e_model.load_state_dict(torch.load('best_end_to_end_enhanced.pth'))
        detailed_results['End-to-End Enhanced'] = self.detailed_evaluation(e2e_model, test_loader, 'end_to_end')
        
        comparison_results = {
            'model_accuracies': results,
            'detailed_results': detailed_results,
            'best_model': max(results, key=results.get),
            'best_accuracy': max(results.values()),
            'improvement_over_baseline': max(results.values()) - 0.2758,
            'dataset_info': {
                'total_samples': len(self.filtered_df),
                'num_words': len(self.word_encoder.classes_),
                'num_speakers': len(self.speaker_encoder.classes_)
            }
        }
        
        with open('enhanced_models_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2, default=str)
        
        print("\n" + "="*80)
        print("Final Enhanced Models Comparison Results:")
        print("="*80)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, accuracy) in enumerate(sorted_results, 1):
            print(f"{i}. {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\nBest model: {comparison_results['best_model']}")
        print(f"Best accuracy: {comparison_results['best_accuracy']:.4f}")
        print(f"Improvement over k-NN: +{comparison_results['improvement_over_baseline']:.4f}")
        
        print("\nBest model performance by speaker:")
        best_model_name = comparison_results['best_model']
        speaker_analysis = detailed_results[best_model_name]['speaker_analysis']
        
        for speaker, data in speaker_analysis.items():
            print(f"  {speaker}: {data['accuracy']:.4f} ({data['total_samples']} samples)")
        
        return comparison_results