import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path
from speech_dataset import SpeechDataset, collate_fn
from speech_cnn_model import SpeechCNN


class SpeechTrainer:
    def __init__(self, model_type='general'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open('data/processed/mappings.json', 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)
        
        self.num_classes = self.mappings['num_words']
        
    def create_dataloaders(self, speaker=None):
        if self.model_type == 'general':
            train_path = 'data/processed/train.csv'
            val_path = 'data/processed/validation.csv'
            test_path = 'data/processed/test.csv'
        else:
            train_path = f'data/processed/speakers/{speaker}/train.csv'
            test_path = f'data/processed/speakers/{speaker}/test.csv'
            val_path = None
        
        train_dataset = SpeechDataset(train_path, self.mappings)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        
        test_dataset = SpeechDataset(test_path, self.mappings)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        val_loader = None
        if val_path and Path(val_path).exists():
            val_dataset = SpeechDataset(val_path, self.mappings)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, speaker=None, epochs=50):
        model = SpeechCNN(self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        train_loader, val_loader, test_loader = self.create_dataloaders(speaker)
        
        train_losses, val_losses = [], []
        best_acc = 0
        
        name = f"{self.model_type}_{speaker}" if speaker else self.model_type
        print(f"Training model {name}...")
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader:
                val_loss, val_acc = self.evaluate(model, val_loader, criterion)
                val_losses.append(val_loss)
                scheduler.step(val_loss)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), f'models/{name}_best.pth')
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}")
        
        # Final test
        test_acc = self.test_model(model, test_loader)
        print(f"Model {name} - Test Accuracy: {test_acc:.3f}")
        
        return model, train_losses, val_losses

    def evaluate(self, model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                total_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(loader), correct / total

    def test_model(self, model, test_loader):
        _, accuracy = self.evaluate(model, test_loader, nn.CrossEntropyLoss())
        return accuracy