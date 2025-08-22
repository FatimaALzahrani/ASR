import torch
import torch.nn as nn


class ImprovedRNN_CNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ImprovedRNN_CNN_Model, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
        )
        
        self.rnn = nn.GRU(256, hidden_dim, num_layers=3, batch_first=True, 
                         dropout=0.3, bidirectional=True)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
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