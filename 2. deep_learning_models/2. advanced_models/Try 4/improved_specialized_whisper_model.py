import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration


class ImprovedSpecializedWhisperModel(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "openai/whisper-small"):
        super().__init__()
        
        print(f"Loading Whisper model: {model_name}")
        
        self.whisper = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        for param in self.whisper.model.encoder.parameters():
            param.requires_grad = False
        
        for param in self.whisper.model.encoder.layers[-3:].parameters():
            param.requires_grad = True
        
        hidden_size = self.whisper.config.d_model
        
        self.feature_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.custom_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.num_classes = num_classes
        print(f"Created specialized model: {num_classes} classes")
    
    def forward(self, input_features):
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)
        
        encoder_outputs = self.whisper.model.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state
        
        adapted_features = self.feature_adapter(hidden_states)
        
        attended_features, attention_weights = self.custom_attention(
            adapted_features, adapted_features, adapted_features
        )
        
        pooled_features = torch.mean(attended_features, dim=1)
        
        logits = self.classifier(pooled_features)
        
        return logits, attention_weights