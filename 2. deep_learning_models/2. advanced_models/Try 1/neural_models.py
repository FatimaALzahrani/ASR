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


class RNN_CNN_Model(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.1),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(128), nn.MaxPool1d(2), nn.Dropout(0.1),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.15),
        )
        
        self.rnn = nn.GRU(256, hidden_dim, num_layers=2, batch_first=True, 
                         dropout=0.2, bidirectional=True)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        x_cnn = x.transpose(1, 2)
        cnn_output = self.cnn(x_cnn)
        cnn_output = cnn_output.transpose(1, 2)
        
        rnn_output, _ = self.rnn(cnn_output)
        
        attention_weights = torch.softmax(self.attention(rnn_output), dim=1)
        attended_output = torch.sum(rnn_output * attention_weights, dim=1)
        
        output = self.classifier(attended_output)
        return output


class EndToEnd_Model(nn.Module):
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.1),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.15),
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512, dropout=0.15, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.global_attention = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(256, 128),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
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