import torch
import torch.nn as nn


class PersonalizedModel(nn.Module):
    def __init__(self, base_model, speaker_embedding_dim=32):
        super(PersonalizedModel, self).__init__()
        
        self.base_model = base_model
        self.speaker_embeddings = nn.Embedding(5, speaker_embedding_dim)
        
        # Get the original classifier's input dimension
        original_classifier = self.base_model.classifier
        
        # Find the last linear layer to get the correct input dimension
        last_linear_layer = None
        for layer in reversed(original_classifier):
            if isinstance(layer, nn.Linear):
                last_linear_layer = layer
                break
        
        if last_linear_layer is None:
            raise ValueError("No Linear layer found in base model classifier")
        
        # Store original output features
        self.num_classes = last_linear_layer.out_features
        audio_feature_dim = last_linear_layer.in_features
        
        # Create new classifier that combines audio features with speaker embeddings
        self.feature_combiner = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(audio_feature_dim + speaker_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes)
        )
        
        # Remove the last linear layer from base model classifier
        classifier_layers = list(original_classifier.children())[:-1]
        self.base_model.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, x, speaker_ids):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Extract audio features using base model (without final classification)
        features = self.base_model.feature_extractor(x)
        features = features.squeeze(-1)
        
        # Apply all classifier layers except the last one
        for layer in self.base_model.classifier:
            features = layer(features)
        
        # Get speaker embeddings
        speaker_emb = self.speaker_embeddings(speaker_ids)
        
        # Combine audio features with speaker embeddings
        combined_features = torch.cat([features, speaker_emb], dim=1)
        
        # Final classification
        output = self.feature_combiner(combined_features)
        
        return output