import torch
import torch.nn as nn


class HMM_DNN_Model(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.BatchNorm1d(hidden_dim // 2), nn.Dropout(0.15),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(), nn.Dropout(0.1)
        )
        
        self.lstm = nn.LSTM(hidden_dim // 4, hidden_dim // 8, num_layers=2, 
                           batch_first=True, dropout=0.1, bidirectional=True)
        
        self.attention = nn.Linear(hidden_dim // 4, 1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_dim // 4, num_classes))
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        x_reshaped = x.view(-1, input_dim)
        dnn_output = self.dnn(x_reshaped)
        dnn_output = dnn_output.view(batch_size, seq_len, -1)
        
        lstm_output, _ = self.lstm(dnn_output)
        
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)
        
        output = self.classifier(attended_output)
        return output