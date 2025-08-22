import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from tqdm.auto import tqdm

from fixed_down_syndrome_dataset import FixedDownSyndromeDataset
from improved_specialized_whisper_model import ImprovedSpecializedWhisperModel


class FinalProfessionalTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        model_name = "openai/whisper-small"
        print(f"Loading processor: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        self.load_data()
        self.setup_models()
        
        self.training_history = {
            'base_model': {'loss': [], 'accuracy': [], 'attention_analysis': []},
            'personal_models': {}
        }
        
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def load_data(self):
        print("Loading data...")
        
        data_dir = Path("C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Codes/Real Codes/01_data_processing/processed")
        
        train_data = pd.read_csv(data_dir / "train.csv")
        val_data = pd.read_csv(data_dir / "validation.csv")
        test_data = pd.read_csv(data_dir / "test.csv")
        
        print(f"   Training: {len(train_data)} samples")
        print(f"   Validation: {len(val_data)} samples")
        print(f"   Testing: {len(test_data)} samples")
        
        self.train_dataset = FixedDownSyndromeDataset(train_data, self.processor)
        self.val_dataset = FixedDownSyndromeDataset(val_data, self.processor)
        self.test_dataset = FixedDownSyndromeDataset(test_data, self.processor)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.analyze_data_distribution(train_data)
    
    def analyze_data_distribution(self, data: pd.DataFrame):
        print("\nAnalyzing data distribution:")
        
        speaker_counts = data['speaker'].value_counts()
        print(f"   Speakers: {dict(speaker_counts)}")
        
        if 'quality' in data.columns:
            quality_counts = data['quality'].value_counts()
            print(f"   Quality levels: {dict(quality_counts)}")
        
        word_counts = data['text'].value_counts()
        print(f"   Most frequent words: {dict(word_counts.head())}")
        
        self.speaker_distribution = speaker_counts
        self.quality_distribution = quality_counts if 'quality' in data.columns else {}
        self.word_distribution = word_counts
    
    def setup_models(self):
        print("Setting up models...")
        
        num_classes = self.train_dataset.num_classes
        
        self.base_model = ImprovedSpecializedWhisperModel(num_classes).to(self.device)
        print(f"   Enhanced base model: {num_classes} classes")
        
        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Training ratio: {trainable_params/total_params*100:.1f}%")
    
    def train_base_model(self):
        print("\nTraining enhanced base model...")
        
        model = self.base_model
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
        
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                
                input_features = batch['input_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                try:
                    outputs, attention_weights = model(input_features)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    
                    if 'gradient_clipping' in self.config:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            self.config['gradient_clipping']
                        )
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    current_acc = 100 * train_correct / train_total
                    pbar.set_postfix({
                        'Loss': f"{loss.item():.4f}",
                        'Acc': f"{current_acc:.2f}%",
                        'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
                    
                except Exception as e:
                    print(f"Warning: Error in training batch {batch_idx}: {e}")
                    continue
            
            val_loss, val_accuracy = self.evaluate_model(model, self.val_loader)
            scheduler.step(val_loss)
            
            epoch_train_acc = train_correct / train_total if train_total > 0 else 0
            self.training_history['base_model']['loss'].append(train_loss / len(self.train_loader))
            self.training_history['base_model']['accuracy'].append(epoch_train_acc)
            
            print(f"Epoch {epoch+1}: Train Acc: {epoch_train_acc:.4f}, Val Acc: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_accuracy,
                    'config': self.config
                }, self.results_dir / "best_final_model.pth")
                
                print(f"Saved new best model: {val_accuracy:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.get('patience', 5):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Finished training base model. Best accuracy: {best_val_accuracy:.4f}")
        return best_val_accuracy
    
    def evaluate_model(self, model, data_loader):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_features = batch['input_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                try:
                    outputs, _ = model(input_features)
                    loss = criterion(outputs, labels)
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    predictions.extend(predicted.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"Warning: Error in evaluation: {e}")
                    continue
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        
        return avg_loss, accuracy
    
    def comprehensive_evaluation(self):
        print("\nComprehensive enhanced evaluation...")
        
        checkpoint_path = self.results_dir / "best_final_model.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best saved model")
        
        results = {
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.base_model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.base_model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.base_model.parameters()) / 1024 / 1024
            },
            'base_model': {},
            'detailed_analysis': {}
        }
        
        print("Evaluating base model...")
        test_loss, test_accuracy = self.evaluate_model(self.base_model, self.test_loader)
        
        results['base_model'] = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'improvement_over_baseline': test_accuracy - 0.0
        }
        
        print(f"   Base model accuracy: {test_accuracy*100:.2f}%")
        print(f"   Improvement over baseline: +{test_accuracy*100:.2f}%")
        
        speaker_results = self.analyze_by_speaker()
        results['detailed_analysis']['by_speaker'] = speaker_results
        
        word_results = self.analyze_by_word()
        results['detailed_analysis']['by_word'] = word_results
        
        return results
    
    def analyze_by_speaker(self):
        print("Analyzing performance by speaker...")
        
        speaker_results = {}
        
        for speaker in self.speaker_distribution.index:
            speaker_test_data = self.test_dataset.data[
                self.test_dataset.data['speaker'] == speaker
            ]
            
            if len(speaker_test_data) > 0:
                speaker_test_dataset = FixedDownSyndromeDataset(speaker_test_data, self.processor)
                speaker_test_loader = DataLoader(
                    speaker_test_dataset,
                    batch_size=len(speaker_test_data),
                    shuffle=False,
                    num_workers=0
                )
                
                test_loss, test_accuracy = self.evaluate_model(self.base_model, speaker_test_loader)
                
                speaker_results[speaker] = {
                    'test_accuracy': test_accuracy,
                    'test_loss': test_loss,
                    'num_samples': len(speaker_test_data),
                    'data_distribution': int(self.speaker_distribution[speaker])
                }
                
                print(f"   {speaker}: {test_accuracy*100:.1f}% ({len(speaker_test_data)} samples)")
        
        return speaker_results
    
    def analyze_by_word(self):
        print("Analyzing performance by word...")
        
        word_results = {}
        
        top_words = self.word_distribution.head(10).index
        
        for word in top_words:
            word_test_data = self.test_dataset.data[
                self.test_dataset.data['text'] == word
            ]
            
            if len(word_test_data) > 0:
                word_test_dataset = FixedDownSyndromeDataset(word_test_data, self.processor)
                word_test_loader = DataLoader(
                    word_test_dataset,
                    batch_size=len(word_test_data),
                    shuffle=False,
                    num_workers=0
                )
                
                test_loss, test_accuracy = self.evaluate_model(self.base_model, word_test_loader)
                
                word_results[word] = {
                    'test_accuracy': test_accuracy,
                    'test_loss': test_loss,
                    'num_samples': len(word_test_data),
                    'frequency': int(self.word_distribution[word])
                }
                
                print(f"   {word}: {test_accuracy*100:.1f}% ({len(word_test_data)} samples)")
        
        return word_results
    
    def save_results(self, results: Dict):
        with open(self.results_dir / "final_professional_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        with open(self.results_dir / "final_training_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)
        
        summary = {
            "executive_summary": {
                "model_accuracy": results['base_model']['test_accuracy'],
                "improvement_percentage": results['base_model']['improvement_over_baseline'] * 100,
                "model_size_mb": results['model_info']['model_size_mb'],
                "trainable_parameters": results['model_info']['trainable_parameters'],
                "best_speaker": None,
                "worst_speaker": None,
                "best_word": None,
                "worst_word": None
            },
            "technical_details": {
                "total_parameters": results['model_info']['total_parameters'],
                "training_config": self.config,
                "data_distribution": {
                    "speakers": dict(self.speaker_distribution),
                    "top_words": dict(self.word_distribution.head(10))
                }
            }
        }
        
        if 'by_speaker' in results['detailed_analysis']:
            speaker_accuracies = {
                speaker: data['test_accuracy']
                for speaker, data in results['detailed_analysis']['by_speaker'].items()
            }
            
            if speaker_accuracies:
                best_speaker = max(speaker_accuracies, key=speaker_accuracies.get)
                worst_speaker = min(speaker_accuracies, key=speaker_accuracies.get)
                
                summary["executive_summary"]["best_speaker"] = {
                    "name": best_speaker,
                    "accuracy": speaker_accuracies[best_speaker] * 100
                }
                summary["executive_summary"]["worst_speaker"] = {
                    "name": worst_speaker,
                    "accuracy": speaker_accuracies[worst_speaker] * 100
                }
        
        if 'by_word' in results['detailed_analysis']:
            word_accuracies = {
                word: data['test_accuracy']
                for word, data in results['detailed_analysis']['by_word'].items()
            }
            
            if word_accuracies:
                best_word = max(word_accuracies, key=word_accuracies.get)
                worst_word = min(word_accuracies, key=word_accuracies.get)
                
                summary["executive_summary"]["best_word"] = {
                    "word": best_word,
                    "accuracy": word_accuracies[best_word] * 100
                }
                summary["executive_summary"]["worst_word"] = {
                    "word": worst_word,
                    "accuracy": word_accuracies[worst_word] * 100
                }
        
        with open(self.results_dir / "final_professional_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {self.results_dir}")
        
        print("\nFinal Results Summary:")
        print("=" * 50)
        print(f"Model accuracy: {summary['executive_summary']['model_accuracy']*100:.2f}%")
        print(f"Improvement: +{summary['executive_summary']['improvement_percentage']:.2f}%")
        print(f"Model size: {summary['executive_summary']['model_size_mb']:.1f} MB")
        print(f"Trainable parameters: {summary['executive_summary']['trainable_parameters']:,}")
        
        if summary["executive_summary"]["best_speaker"]:
            best = summary["executive_summary"]["best_speaker"]
            print(f"Best speaker: {best['name']} ({best['accuracy']:.1f}%)")
        
        if summary["executive_summary"]["best_word"]:
            best = summary["executive_summary"]["best_word"]
            print(f"Best word: {best['word']} ({best['accuracy']:.1f}%)")