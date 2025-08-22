import torch
import torch.nn as nn


class SimpleAudioClassifier(nn.Module):
    def __init__(self, num_classes, input_size=16000):
        super(SimpleAudioClassifier, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=16),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(4),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        features = self.feature_extractor(x)
        features = features.squeeze(-1)
        
        # Ensure features have correct shape
        if len(features.shape) != 2:
            features = features.view(features.size(0), -1)
        
        output = self.classifier(features)
        
        return output