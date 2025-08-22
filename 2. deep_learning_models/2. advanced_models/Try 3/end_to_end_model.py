import torch
import torch.nn as nn


class ImprovedEndToEndModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ImprovedEndToEndModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, 
            dropout=0.2, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.global_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        x_encoded = x.transpose(1, 2)
        encoded_features = self.encoder(x_encoded)
        
        encoded_features = encoded_features.transpose(1, 2)
        encoded_features = encoded_features.transpose(0, 1)
        transformer_output = self.transformer(encoded_features)
        transformer_output = transformer_output.transpose(0, 1)
        
        attention_weights = torch.softmax(self.global_attention(transformer_output), dim=1)
        attended_features = torch.sum(transformer_output * attention_weights, dim=1)
        
        output = self.classifier(attended_features)
        
        return output