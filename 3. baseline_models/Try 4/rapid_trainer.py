import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from audio_dataset import AudioDataset
from simple_audio_classifier import SimpleAudioClassifier
from personalized_model import PersonalizedModel


class RapidTrainer:
    def __init__(self, data_path="processed_dataset.csv"):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.filtered_df = None
        
        print(f"Using device: {self.device}")
        self.load_data()
        self.setup_encoders()
        
    def load_data(self):
        print("Loading data...")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} samples")
        print(f"{self.df['word'].nunique()} unique words")
        print(f"{self.df['speaker'].nunique()} speakers")
        
    def setup_encoders(self):
        print("Setting up encoders...")
        
        # Filter words with at least 2 samples for training
        word_counts = self.df['word'].value_counts()
        valid_words = word_counts[word_counts >= 2].index
        self.filtered_df = self.df[self.df['word'].isin(valid_words)]
        
        self.word_encoder = LabelEncoder()
        self.word_encoder.fit(self.filtered_df['word'])
        
        self.speaker_encoder = LabelEncoder()
        self.speaker_encoder.fit(self.filtered_df['speaker'])
        
        print(f"Original data: {len(self.df)} samples, {self.df['word'].nunique()} words")
        print(f"Filtered data: {len(self.filtered_df)} samples, {len(valid_words)} words")
        print(f"Word encoder setup for {len(self.word_encoder.classes_)} words")
        print(f"Speaker encoder setup for {len(self.speaker_encoder.classes_)} speakers")
    
    def create_data_loaders(self, test_size=0.2, batch_size=32):
        print("Creating data loaders...")
        
        # Check if we have actual audio files or need to simulate
        has_audio_files = False
        if 'file_path' in self.filtered_df.columns:
            sample_paths = self.filtered_df['file_path'].dropna().head(5)
            has_audio_files = any(os.path.exists(path) for path in sample_paths if isinstance(path, str))
        
        if not has_audio_files:
            print("Warning: No valid audio files found. Using simulated audio data for testing.")
        
        # Use the filtered dataframe from setup_encoders
        if len(self.filtered_df) < 10:
            print("Warning: Very few samples available. Using simple split without stratification.")
            train_df, test_df = train_test_split(
                self.filtered_df, test_size=test_size, random_state=42
            )
        else:
            train_df, test_df = train_test_split(
                self.filtered_df, test_size=test_size, 
                stratify=self.filtered_df['word'], random_state=42
            )
        
        train_dataset = AudioDataset(train_df, self.word_encoder, augment=True)
        test_dataset = AudioDataset(test_df, self.word_encoder, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training data: {len(train_dataset)} samples")
        print(f"Testing data: {len(test_dataset)} samples")
        
        return train_loader, test_loader
    
    def train_basic_model(self, epochs=10, lr=0.001):
        print("Training basic model...")
        
        train_loader, test_loader = self.create_data_loaders()
        
        model = SimpleAudioClassifier(num_classes=len(self.word_encoder.classes_))
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        best_accuracy = 0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_idx, (audio, labels, speakers) in enumerate(train_loader):
                audio, labels = audio.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(audio)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            accuracy = self.evaluate_model(model, test_loader)
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Accuracy={accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'best_basic_model.pth')
            
            scheduler.step()
        
        self.results['basic_model'] = {
            'accuracy': best_accuracy,
            'epochs': epochs,
            'model_type': 'basic'
        }
        
        print(f"Best basic model accuracy: {best_accuracy:.4f}")
        return model, best_accuracy
    
    def train_personalized_model(self, base_model, epochs=15, lr=0.0005):
        print("Training personalized model...")
        
        train_loader, test_loader = self.create_data_loaders()
        
        model = PersonalizedModel(base_model)
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
        
        best_accuracy = 0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_idx, (audio, labels, speakers) in enumerate(train_loader):
                audio, labels = audio.to(self.device), labels.to(self.device)
                speaker_ids = torch.LongTensor([self.speaker_encoder.transform([s])[0] for s in speakers]).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(audio, speaker_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            accuracy = self.evaluate_personalized_model(model, test_loader)
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Accuracy={accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'best_personalized_model.pth')
            
            scheduler.step()
        
        self.results['personalized_model'] = {
            'accuracy': best_accuracy,
            'epochs': epochs,
            'model_type': 'personalized'
        }
        
        print(f"Best personalized model accuracy: {best_accuracy:.4f}")
        return model, best_accuracy
    
    def evaluate_model(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for audio, labels, speakers in test_loader:
                audio, labels = audio.to(self.device), labels.to(self.device)
                outputs = model(audio)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def evaluate_personalized_model(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for audio, labels, speakers in test_loader:
                audio, labels = audio.to(self.device), labels.to(self.device)
                speaker_ids = torch.LongTensor([self.speaker_encoder.transform([s])[0] for s in speakers]).to(self.device)
                outputs = model(audio, speaker_ids)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def detailed_evaluation(self, model, model_type='basic'):
        print(f"Detailed evaluation for {model_type} model...")
        
        _, test_loader = self.create_data_loaders()
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_speakers = []
        
        with torch.no_grad():
            for audio, labels, speakers in test_loader:
                audio, labels = audio.to(self.device), labels.to(self.device)
                
                if model_type == 'personalized':
                    speaker_ids = torch.LongTensor([self.speaker_encoder.transform([s])[0] for s in speakers]).to(self.device)
                    outputs = model(audio, speaker_ids)
                else:
                    outputs = model(audio)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_speakers.extend(speakers)
        
        predicted_words = self.word_encoder.inverse_transform(all_predictions)
        actual_words = self.word_encoder.inverse_transform(all_labels)
        
        overall_accuracy = accuracy_score(all_labels, all_predictions)
        
        speaker_analysis = {}
        for speaker in set(all_speakers):
            speaker_mask = [s == speaker for s in all_speakers]
            speaker_predictions = [all_predictions[i] for i, mask in enumerate(speaker_mask) if mask]
            speaker_labels = [all_labels[i] for i, mask in enumerate(speaker_mask) if mask]
            
            if speaker_predictions:
                speaker_accuracy = accuracy_score(speaker_labels, speaker_predictions)
                speaker_analysis[speaker] = {
                    'accuracy': speaker_accuracy,
                    'total_samples': len(speaker_predictions),
                    'correct_predictions': sum(1 for p, l in zip(speaker_predictions, speaker_labels) if p == l)
                }
        
        word_analysis = {}
        for word_idx, word in enumerate(self.word_encoder.classes_):
            word_mask = [l == word_idx for l in all_labels]
            word_predictions = [all_predictions[i] for i, mask in enumerate(word_mask) if mask]
            word_labels = [all_labels[i] for i, mask in enumerate(word_mask) if mask]
            
            if word_predictions:
                word_accuracy = accuracy_score(word_labels, word_predictions)
                word_analysis[word] = {
                    'accuracy': word_accuracy,
                    'total_samples': len(word_predictions),
                    'correct_predictions': sum(1 for p, l in zip(word_predictions, word_labels) if p == l)
                }
        
        detailed_results = {
            'model_type': model_type,
            'overall_accuracy': overall_accuracy,
            'speaker_analysis': speaker_analysis,
            'word_analysis': word_analysis,
            'total_samples': len(all_predictions),
            'correct_predictions': sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
        }
        
        with open(f'detailed_results_{model_type}.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        print(f"Detailed evaluation saved to: detailed_results_{model_type}.json")
        
        return detailed_results
    
    def run_rapid_training(self):
        print("Starting rapid model training...")
        print("=" * 60)
        
        basic_model, basic_accuracy = self.train_basic_model(epochs=8)
        basic_results = self.detailed_evaluation(basic_model, 'basic')
        
        print("\n" + "=" * 60)
        
        personalized_model, personalized_accuracy = self.train_personalized_model(basic_model, epochs=12)
        personalized_results = self.detailed_evaluation(personalized_model, 'personalized')
        
        summary = {
            'training_completed': True,
            'basic_model': {
                'accuracy': basic_accuracy,
                'detailed_results': basic_results
            },
            'personalized_model': {
                'accuracy': personalized_accuracy,
                'detailed_results': personalized_results
            },
            'improvement': personalized_accuracy - basic_accuracy,
            'improvement_percentage': ((personalized_accuracy - basic_accuracy) / basic_accuracy) * 100
        }
        
        with open('rapid_training_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print("Rapid training completed successfully!")
        print(f"Basic model: {basic_accuracy:.4f}")
        print(f"Personalized model: {personalized_accuracy:.4f}")
        print(f"Improvement: +{summary['improvement']:.4f} ({summary['improvement_percentage']:.1f}%)")
        
        return summary