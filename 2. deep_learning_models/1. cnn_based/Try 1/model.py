import torch
import torch.nn as nn


class SpeechClassifier(nn.Module):
    
    def __init__(self, num_classes, input_dim=53, dropout=0.3):
        super().__init__()
        
        # CNN layers for feature extraction
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv1d(input_dim, 128, 5, padding=2),
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            
            # Second conv block
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            
            # Third conv block
            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            # First FC layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            
            # Second FC layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Transpose for Conv1d (batch, features, time)
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = self.cnn(x)
        
        # Apply classification layers
        return self.classifier(x)