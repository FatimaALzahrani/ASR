import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


class AttentionBlock(layers.Layer):
    def __init__(self, units):
        super(AttentionBlock, self).__init__()
        self.units = units
        self.W = layers.Dense(units)
        self.U = layers.Dense(units)
        self.V = layers.Dense(1)
        
    def call(self, inputs):
        score = tf.nn.tanh(self.W(inputs) + self.U(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        return tf.reduce_sum(context_vector, axis=1)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ConformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, conv_kernel_size=31):
        super(ConformerBlock, self).__init__()
        
        self.ff1 = keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(ff_dim, activation='swish'),
            layers.Dropout(0.1),
            layers.Dense(embed_dim)
        ])
        
        self.mhsa = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.norm_mhsa = layers.LayerNormalization()
        
        self.conv = keras.Sequential([
            layers.LayerNormalization(),
            layers.Conv1D(embed_dim * 2, 1, activation='glu'),
            layers.Conv1D(embed_dim, conv_kernel_size, padding='same', groups=embed_dim),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            layers.Conv1D(embed_dim, 1),
            layers.Dropout(0.1)
        ])
        
        self.ff2 = keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(ff_dim, activation='swish'),
            layers.Dropout(0.1),
            layers.Dense(embed_dim)
        ])
        
        self.norm_final = layers.LayerNormalization()
        
    def call(self, inputs, training=None):
        x = inputs + 0.5 * self.ff1(inputs)
        
        attn_output = self.mhsa(x, x, training=training)
        x = self.norm_mhsa(x + attn_output)
        
        conv_output = self.conv(x, training=training)
        x = x + conv_output
        
        x = x + 0.5 * self.ff2(x)
        
        return self.norm_final(x)


class WaveNetBlock(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate):
        super(WaveNetBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
        self.conv_tanh = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, 
                                      padding='causal', activation='tanh')
        self.conv_sigmoid = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, 
                                         padding='causal', activation='sigmoid')
        self.conv_1x1 = layers.Conv1D(filters, 1)
        
    def call(self, inputs):
        tanh_out = self.conv_tanh(inputs)
        sigmoid_out = self.conv_sigmoid(inputs)
        gated = tanh_out * sigmoid_out
        
        output = self.conv_1x1(gated)
        return output, gated


class PyTorchCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PyTorchCNN, self).__init__()
        
        self.reshape_layer = nn.Linear(input_size, 128 * 16)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.4),
            
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.reshape_layer(x)
        x = x.view(x.size(0), 128, 16)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x


class DeepModelFactory:
    @staticmethod
    def create_advanced_cnn(input_shape, num_classes):
        model = models.Sequential([
            layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),
            
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(512, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(512, 3, activation='relu', padding='same'),
            layers.GlobalMaxPooling1D(),
            
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model

    @staticmethod
    def create_lstm_attention(input_shape, num_classes):
        inputs = layers.Input(shape=(input_shape,))
        
        sequence_length = min(50, input_shape // 10)
        features_per_step = input_shape // sequence_length
        
        if input_shape % sequence_length != 0:
            target_size = sequence_length * features_per_step
            x = layers.Dense(target_size)(inputs)
        else:
            x = inputs
        
        x = layers.Reshape((sequence_length, features_per_step))(x)
        
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
        
        attention_output = AttentionBlock(64)(x)
        
        x = layers.Dense(512, activation='relu')(attention_output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model

    @staticmethod
    def create_transformer(input_shape, num_classes):
        inputs = layers.Input(shape=(input_shape,))
        
        embed_dim = 256
        
        if input_shape >= embed_dim:
            seq_len = input_shape // embed_dim
            if input_shape % embed_dim != 0:
                target_size = seq_len * embed_dim
                x = layers.Dense(target_size)(inputs)
            else:
                x = inputs
        else:
            seq_len = 4
            target_size = seq_len * embed_dim
            x = layers.Dense(target_size)(inputs)
        
        x = layers.Reshape((seq_len, embed_dim))(x)
        
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embedding = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)(positions)
        x = x + position_embedding
        
        for _ in range(4):
            x = TransformerBlock(embed_dim, num_heads=8, ff_dim=512)(x, training=True)
        
        x = layers.GlobalAveragePooling1D()(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model

    @staticmethod
    def create_hybrid_resnet_lstm(input_shape, num_classes):
        inputs = layers.Input(shape=(input_shape,))
        
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        for i in range(3):
            residual = x
            x = layers.Dense(512, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, residual])
        
        x = layers.Reshape((32, 16))(x)
        
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(64, dropout=0.3))(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model

    @staticmethod
    def create_autoencoder_classifier(input_shape, num_classes):
        inputs = layers.Input(shape=(input_shape,))
        
        encoded = layers.Dense(1024, activation='relu')(inputs)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.3)(encoded)
        
        encoded = layers.Dense(512, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.3)(encoded)
        
        encoded = layers.Dense(256, activation='relu')(encoded)
        latent = layers.BatchNormalization()(encoded)
        
        decoded = layers.Dense(512, activation='relu')(latent)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.3)(decoded)
        
        decoded = layers.Dense(1024, activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.3)(decoded)
        
        decoded = layers.Dense(input_shape, activation='linear')(decoded)
        
        classifier = layers.Dense(512, activation='relu')(latent)
        classifier = layers.BatchNormalization()(classifier)
        classifier = layers.Dropout(0.4)(classifier)
        
        classifier = layers.Dense(256, activation='relu')(classifier)
        classifier = layers.Dropout(0.3)(classifier)
        
        outputs = layers.Dense(num_classes, activation='softmax')(classifier)
        
        autoencoder = models.Model(inputs, decoded)
        classifier_model = models.Model(inputs, outputs)
        
        return autoencoder, classifier_model

    @staticmethod
    def create_conformer(input_shape, num_classes):
        inputs = layers.Input(shape=(input_shape,))
        
        x = layers.Reshape((input_shape // 16, 16))(inputs)
        
        embed_dim = 256
        x = layers.Dense(embed_dim)(x)
        
        for _ in range(4):
            x = ConformerBlock(embed_dim, num_heads=8, ff_dim=embed_dim*4)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(512, activation='swish')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='swish')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model

    @staticmethod
    def create_wavenet(input_shape, num_classes):
        inputs = layers.Input(shape=(input_shape,))
        x = layers.Reshape((input_shape, 1))(inputs)
        
        x = layers.Conv1D(64, 2, padding='causal')(x)
        
        skip_connections = []
        
        dilation_rates = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        for dilation_rate in dilation_rates:
            wavenet_layer = WaveNetBlock(64, 2, dilation_rate)
            residual, skip = wavenet_layer(x)
            
            x = layers.Add()([x, residual])
            skip_connections.append(skip)
        
        x = layers.Add()(skip_connections)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(128, 1, activation='relu')(x)
        x = layers.Conv1D(256, 1, activation='relu')(x)
        
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model

    @staticmethod
    def create_capsule_network(input_shape, num_classes):
        inputs = layers.Input(shape=(input_shape,))
        
        x = layers.Reshape((input_shape, 1))(inputs)
        x = layers.Conv1D(256, 9, activation='relu')(x)
        
        primary_caps = layers.Conv1D(32 * 8, 9, strides=2)(x)
        primary_caps = layers.Reshape((-1, 8))(primary_caps)
        primary_caps = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))(primary_caps)
        
        digital_caps = layers.Dense(num_classes * 16)(primary_caps)
        digital_caps = layers.Reshape((num_classes, 16))(digital_caps)
        
        capsule_lengths = layers.Lambda(
            lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1))
        )(digital_caps)
        
        outputs = layers.Activation('softmax')(capsule_lengths)
        
        model = models.Model(inputs, outputs)
        return model

    @staticmethod
    def train_pytorch_cnn(X_train, X_test, y_train, y_test, num_classes):
        print("Training PyTorch CNN...")
        
        X_train_torch = torch.FloatTensor(X_train)
        X_test_torch = torch.FloatTensor(X_test)
        y_train_torch = torch.LongTensor(y_train)
        y_test_torch = torch.LongTensor(y_test)
        
        train_dataset = TensorDataset(X_train_torch, y_train_torch)
        test_dataset = TensorDataset(X_test_torch, y_test_torch)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PyTorchCNN(X_train.shape[1], num_classes).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        model.train()
        for epoch in range(50):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {running_loss/len(train_loader):.4f}')
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        print(f"PyTorch CNN Accuracy: {accuracy:.4f}")
        
        return accuracy