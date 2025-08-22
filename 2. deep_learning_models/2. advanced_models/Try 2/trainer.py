import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class ModelTrainer:
    
    def __init__(self, device, label_encoder, word_to_id):
        self.device = device
        self.label_encoder = label_encoder
        self.word_to_id = word_to_id
    
    def train_single_model(self, train_data, test_data, model_name):
        try:
            from audio_dataset import AudioDataset
            from models import HMM_DNN_Model
            
            train_dataset = AudioDataset(train_data, feature_type='mfcc', augment=True)
            test_dataset = AudioDataset(test_data, feature_type='mfcc', augment=False)
            
            train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, 
                                     collate_fn=self.collate_fn, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, 
                                    collate_fn=self.collate_fn, num_workers=0)
            
            model = HMM_DNN_Model(input_dim=39, hidden_dim=512, 
                                 num_classes=len(self.label_encoder.classes_)).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            
            for epoch in range(15):
                model.train()
                total_loss = 0
                
                for features, labels, _ in train_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                scheduler.step()
            
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, labels, _ in test_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            return accuracy
            
        except Exception as e:
            print(f"Training error: {e}")
            return 0.0
    
    def train_full_data_model(self):
        try:
            from audio_dataset import AudioDataset
            from models import HMM_DNN_Model
            
            full_dataset = AudioDataset(self.all_data, feature_type='mfcc', augment=True)
            full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, 
                                   collate_fn=self.collate_fn, num_workers=0)
            
            model = HMM_DNN_Model(input_dim=39, hidden_dim=256,
                                 num_classes=len(self.label_encoder.classes_)).to(self.device)
            
            criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
            optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
            
            print(f"Training with strong regularization...")
            
            for epoch in range(30):
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                for features, labels, _ in full_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                if epoch % 10 == 0:
                    train_acc = correct / total
                    print(f"  Epoch {epoch}: Loss={total_loss/len(full_loader):.4f}, Train Acc={train_acc:.4f}")
            
            final_accuracy = correct / total
            print(f"Final training accuracy: {final_accuracy:.4f}")
            
            torch.save(model.state_dict(), 'full_data_model.pth')
            print(f"Model saved: full_data_model.pth")
            
            return final_accuracy
            
        except Exception as e:
            print(f"Error: {e}")
            return 0.0
    
    def collate_fn(self, batch):
        features, words, speakers = zip(*batch)
        features_tensor = torch.stack(features)
        word_ids = [self.word_to_id[word] for word in words]
        labels_tensor = torch.tensor(word_ids)
        return features_tensor, labels_tensor, speakers
    
    def set_all_data(self, all_data):
        self.all_data = all_data