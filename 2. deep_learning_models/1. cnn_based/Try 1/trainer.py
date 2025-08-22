import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path

from audio_processor import AudioProcessor
from dataset import SpeechDataset, collate_fn
from model import SpeechClassifier
from evaluator import ModelEvaluator


class SpeechTrainer:
    
    def __init__(self, model_type='general'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load mappings
        with open('output_files/processed_data/mappings.json', 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)
        
        self.num_classes = self.mappings['num_words']
        self.audio_processor = AudioProcessor()
        self.evaluator = ModelEvaluator(self.device)
        
        print(f"Using device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
        
    def create_dataloaders(self, speaker=None, batch_size=16):
        if self.model_type == 'general':
            train_path = 'output_files/processed_data/train.csv'
            val_path = 'output_files/processed_data/validation.csv'
            test_path = 'output_files/processed_data/test.csv'
        else:
            train_path = f'output_files/processed_data/speakers/{speaker}/train.csv'
            test_path = f'output_files/processed_data/speakers/{speaker}/test.csv'
            val_path = None
            
        # Training dataset with augmentation
        train_dataset = SpeechDataset(
            train_path, 
            self.mappings, 
            audio_processor=self.audio_processor,
            augment=True
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=0
        )
        
        # Test dataset without augmentation
        test_dataset = SpeechDataset(
            test_path, 
            self.mappings, 
            audio_processor=self.audio_processor,
            augment=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=0
        )
        
        # Validation dataset (if exists)
        val_loader = None
        if val_path and Path(val_path).exists():
            val_dataset = SpeechDataset(
                val_path, 
                self.mappings, 
                audio_processor=self.audio_processor,
                augment=False
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=collate_fn, 
                num_workers=0
            )
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, speaker=None, epochs=80, lr=0.0005):
        # Initialize model
        model = SpeechClassifier(self.num_classes, dropout=0.4).to(self.device)
        
        # Setup training components
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_dataloaders(speaker)
        
        # Training tracking
        train_losses, val_losses, val_accs = [], [], []
        best_acc = 0
        patience_counter = 0
        patience = 15
        
        # Model name for saving
        name = f"{self.model_type}_{speaker}" if speaker else self.model_type
        print(f"Training model: {name}")
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            correct_train, total_train = 0, 0
            
            # Training phase
            for data, target, _ in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                
                # Calculate training accuracy
                pred = output.argmax(dim=1)
                correct_train += pred.eq(target).sum().item()
                total_train += target.size(0)
            
            scheduler.step()
            avg_train_loss = train_loss / len(train_loader)
            train_acc = correct_train / total_train
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss, val_acc = 0, 0
            if val_loader:
                val_loss, val_acc = self.evaluator.evaluate_model(
                    model, val_loader, criterion
                )
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), f'models/{name}_best.pth')
                else:
                    patience_counter += 1
                
                # Print progress
                if epoch % 10 == 0:
                    lr_current = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch}: Train Acc: {train_acc:.3f}, "
                          f"Val Acc: {val_acc:.3f}, LR: {lr_current:.2e}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                # No validation set
                if epoch % 10 == 0:
                    lr_current = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch}: Train Acc: {train_acc:.3f}, "
                          f"LR: {lr_current:.2e}")
        
        # Load best model for testing
        if val_loader and Path(f'models/{name}_best.pth').exists():
            model.load_state_dict(torch.load(f'models/{name}_best.pth'))
        
        # Final testing
        test_acc, test_report = self.evaluator.detailed_test(
            model, test_loader, name
        )
        
        return model, train_losses, val_losses, test_report