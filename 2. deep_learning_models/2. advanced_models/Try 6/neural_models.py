import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

class ASRModelArchitectures:
    
    @staticmethod
    def create_enhanced_cnn(input_shape: int, num_classes: int) -> keras.Model:
        inputs = layers.Input(shape=(input_shape,))
        
        x = layers.Reshape((input_shape, 1))(inputs)
        
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs, name='Enhanced_CNN')
        return model
    
    @staticmethod
    def create_enhanced_lstm(input_shape: int, num_classes: int) -> keras.Model:
        inputs = layers.Input(shape=(input_shape,))
        
        time_steps = 20
        features_per_step = input_shape // time_steps
        if input_shape % time_steps != 0:
            target_size = time_steps * features_per_step
            x = layers.Dense(target_size)(inputs)
        else:
            x = inputs
        
        x = layers.Reshape((time_steps, features_per_step))(x)
        
        x = layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
        x = layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs, name='Enhanced_LSTM')
        return model
    
    @staticmethod
    def create_deep_dnn(input_shape: int, num_classes: int) -> keras.Model:
        inputs = layers.Input(shape=(input_shape,))
        
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs, name='Deep_DNN')
        return model
    
    @staticmethod
    def create_hybrid_model(input_shape: int, num_classes: int) -> keras.Model:
        inputs = layers.Input(shape=(input_shape,))
        
        x = layers.BatchNormalization()(inputs)
        
        cnn_branch = layers.Reshape((input_shape, 1))(x)
        cnn_branch = layers.Conv1D(64, 3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = layers.BatchNormalization()(cnn_branch)
        cnn_branch = layers.Dropout(0.3)(cnn_branch)
        
        cnn_branch = layers.Conv1D(128, 3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = layers.BatchNormalization()(cnn_branch)
        cnn_branch = layers.MaxPooling1D(2)(cnn_branch)
        cnn_branch = layers.Dropout(0.4)(cnn_branch)
        
        cnn_branch = layers.Conv1D(256, 3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = layers.BatchNormalization()(cnn_branch)
        cnn_branch = layers.GlobalAveragePooling1D()(cnn_branch)
        
        dnn_branch = layers.Dense(1024, activation='relu')(x)
        dnn_branch = layers.BatchNormalization()(dnn_branch)
        dnn_branch = layers.Dropout(0.4)(dnn_branch)
        
        dnn_branch = layers.Dense(512, activation='relu')(dnn_branch)
        dnn_branch = layers.BatchNormalization()(dnn_branch)
        dnn_branch = layers.Dropout(0.3)(dnn_branch)
        
        merged = layers.Concatenate()([cnn_branch, dnn_branch])
        
        x = layers.Dense(512, activation='relu')(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs, name='Hybrid_CNN_DNN')
        return model
    
    @staticmethod
    def create_transformer_inspired(input_shape: int, num_classes: int) -> keras.Model:
        inputs = layers.Input(shape=(input_shape,))
        
        seq_len = 32
        d_model = max(1, input_shape // seq_len)
        if d_model == 0:
            d_model = 8
        
        target_size = seq_len * d_model
        x = layers.Dense(target_size)(inputs)
        
        x = layers.Reshape((seq_len, d_model))(x)
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        attn_output = layers.MultiHeadAttention(
            num_heads=min(8, d_model), key_dim=max(1, d_model//8)
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        ffn = keras.Sequential([
            layers.Dense(d_model * 4, activation="relu"),
            layers.Dense(d_model),
        ])
        ffn_output = ffn(x)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs, name='Transformer_Inspired')
        return model
    
    @staticmethod
    def create_conformer_inspired(input_shape: int, num_classes: int) -> keras.Model:
        inputs = layers.Input(shape=(input_shape,))
        
        seq_len = 25
        d_model = max(1, input_shape // seq_len)
        if d_model == 0:
            d_model = 8
        
        target_size = seq_len * d_model
        x = layers.Dense(target_size)(inputs)
        
        x = layers.Reshape((seq_len, d_model))(x)
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        ffn1 = layers.Dense(d_model * 4, activation='swish')(x)
        ffn1 = layers.Dense(d_model)(ffn1)
        x = layers.Add()([x, ffn1 * 0.5])
        
        attn_output = layers.MultiHeadAttention(
            num_heads=min(4, d_model), key_dim=max(1, d_model//4)
        )(x, x)
        x = layers.Add()([x, attn_output])
        
        conv_input = layers.LayerNormalization(epsilon=1e-6)(x)
        conv_output = layers.Conv1D(d_model * 2, 31, padding='same', activation='swish')(conv_input)
        conv_output = layers.Conv1D(d_model, 1)(conv_output)
        x = layers.Add()([x, conv_output])
        
        ffn2 = layers.Dense(d_model * 4, activation='swish')(x)
        ffn2 = layers.Dense(d_model)(ffn2)
        x = layers.Add()([x, ffn2 * 0.5])
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(512, activation='swish')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256, activation='swish')(x)
        x = layers.Dropout(0.1)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs, name='Conformer_Inspired')
        return model