import torch
import torch.nn as nn


class SpeechCNN(nn.Module):
    def __init__(self, num_classes, input_dim=13):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, padding=1),
            nn.ReLU(), 
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(), 
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # (batch_size, features, seq_len)
        x = self.cnn(x)
        return self.classifier(x)