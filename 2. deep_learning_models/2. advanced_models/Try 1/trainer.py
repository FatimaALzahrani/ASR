import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from fixed_audio_dataset import FixedAudioDataset
from neural_models import HMM_DNN_Model, RNN_CNN_Model, EndToEnd_Model


class ModelTrainer:
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.load_data()
        self.setup_encoders()
    
    def load_data(self):
        print("Loading processed data...")
        
        try:
            self.train_df = pd.read_csv("processed/train.csv", encoding='utf-8')
            self.val_df = pd.read_csv("processed/validation.csv", encoding='utf-8')
            self.test_df = pd.read_csv("processed/test.csv", encoding='utf-8')
            
            print(f"Train: {len(self.train_df)} samples")
            print(f"Validation: {len(self.val_df)} samples")
            print(f"Test: {len(self.test_df)} samples")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def setup_encoders(self):
        all_words = pd.concat([self.train_df, self.val_df, self.test_df])['word'].unique()
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_words)
        
        self.word_to_id = {word: idx for idx, word in enumerate(self.label_encoder.classes_)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        
        print(f"Encoder setup for {len(self.label_encoder.classes_)} words")
    
    def create_data_loaders(self, feature_type='mfcc', batch_size=32):
        print(f"Creating data loaders for features: {feature_type}")
        
        train_dataset = FixedAudioDataset(self.train_df, feature_type=feature_type, augment=True)
        val_dataset = FixedAudioDataset(self.val_df, feature_type=feature_type, augment=False)
        test_dataset = FixedAudioDataset(self.test_df, feature_type=feature_type, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 collate_fn=self.collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               collate_fn=self.collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                collate_fn=self.collate_fn, num_workers=0)
        
        return train_loader, val_loader, test_loader
    
    def collate_fn(self, batch):
        features, words, speakers = zip(*batch)
        
        features_tensor = torch.stack(features)
        
        word_ids = [self.word_to_id[word] for word in words]
        labels_tensor = torch.tensor(word_ids)
        
        return features_tensor, labels_tensor, speakers
    
    def train_model(self, model, train_loader, val_loader, model_name, epochs=30, lr=0.001):
        print(f"Training model {model_name}...")
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
        
        best_accuracy = 0
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0
            
            for features, labels, _ in train_loader:
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
            
            val_accuracy = self.evaluate_model(model, val_loader)
            train_accuracy = train_correct / train_total
            avg_loss = total_loss / len(train_loader)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}, Loss={avg_loss:.4f}")
            
            scheduler.step()
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), f'fixed_{model_name}_best.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Best accuracy for {model_name}: {best_accuracy:.4f}")
        return best_accuracy
    
    def evaluate_model(self, model, data_loader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels, _ in data_loader:
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
        
        print(f"\nPerformance of {model_name} by speaker:")
        speaker_results = {}
        for speaker in set(all_speakers):
            speaker_mask = [s == speaker for s in all_speakers]
            speaker_preds = [all_predictions[i] for i, mask in enumerate(speaker_mask) if mask]
            speaker_labels = [all_labels[i] for i, mask in enumerate(speaker_mask) if mask]
            
            if speaker_preds:
                speaker_acc = accuracy_score(speaker_labels, speaker_preds)
                speaker_results[speaker] = speaker_acc
                print(f"  {speaker}: {speaker_acc:.4f} ({len(speaker_preds)} samples)")
        
        return accuracy, speaker_results