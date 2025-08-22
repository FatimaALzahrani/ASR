import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deep_models import DeepModelFactory
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.history = {}
        self.model_factory = DeepModelFactory()
        
    def compile_model(self, model, learning_rate=0.001, weight_decay=0.01):
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def get_callbacks(self, patience_early=15, patience_lr=8):
        return [
            callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=patience_early, 
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=patience_lr, 
                min_lr=1e-7
            )
        ]
    
    def train_deep_models(self, X, y, speakers):
        print("Training deep learning models...")
        print("="*70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        num_classes = len(np.unique(y))
        input_shape = X_train.shape[1]
        
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        results = {}
        
        print("\n1. Training Advanced CNN...")
        try:
            cnn_model = self.model_factory.create_advanced_cnn(input_shape, num_classes)
            cnn_model = self.compile_model(cnn_model)
            
            history_cnn = cnn_model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=100,
                batch_size=32,
                callbacks=self.get_callbacks(),
                verbose=0
            )
            
            _, cnn_accuracy = cnn_model.evaluate(X_test, y_test_cat, verbose=0)
            results['Advanced_CNN'] = cnn_accuracy
            self.models['Advanced_CNN'] = cnn_model
            self.history['Advanced_CNN'] = history_cnn
            print(f"Advanced CNN: {cnn_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in CNN: {e}")
            results['Advanced_CNN'] = 0
        
        print("\n2. Training LSTM with Attention...")
        try:
            lstm_model = self.model_factory.create_lstm_attention(input_shape, num_classes)
            lstm_model = self.compile_model(lstm_model)
            
            history_lstm = lstm_model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=80,
                batch_size=32,
                callbacks=self.get_callbacks(),
                verbose=0
            )
            
            _, lstm_accuracy = lstm_model.evaluate(X_test, y_test_cat, verbose=0)
            results['LSTM_Attention'] = lstm_accuracy
            self.models['LSTM_Attention'] = lstm_model
            self.history['LSTM_Attention'] = history_lstm
            print(f"LSTM with Attention: {lstm_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in LSTM: {e}")
            results['LSTM_Attention'] = 0
        
        print("\n3. Training Transformer...")
        try:
            transformer_model = self.model_factory.create_transformer(input_shape, num_classes)
            transformer_model = self.compile_model(transformer_model, learning_rate=0.0005)
            
            history_transformer = transformer_model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=60,
                batch_size=16,
                callbacks=self.get_callbacks(),
                verbose=0
            )
            
            _, transformer_accuracy = transformer_model.evaluate(X_test, y_test_cat, verbose=0)
            results['Transformer'] = transformer_accuracy
            self.models['Transformer'] = transformer_model
            self.history['Transformer'] = history_transformer
            print(f"Transformer: {transformer_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in Transformer: {e}")
            results['Transformer'] = 0
        
        print("\n4. Training Hybrid ResNet-LSTM...")
        try:
            hybrid_model = self.model_factory.create_hybrid_resnet_lstm(input_shape, num_classes)
            hybrid_model = self.compile_model(hybrid_model)
            
            history_hybrid = hybrid_model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=80,
                batch_size=32,
                callbacks=self.get_callbacks(),
                verbose=0
            )
            
            _, hybrid_accuracy = hybrid_model.evaluate(X_test, y_test_cat, verbose=0)
            results['Hybrid_ResNet_LSTM'] = hybrid_accuracy
            self.models['Hybrid_ResNet_LSTM'] = hybrid_model
            self.history['Hybrid_ResNet_LSTM'] = history_hybrid
            print(f"Hybrid ResNet-LSTM: {hybrid_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in Hybrid model: {e}")
            results['Hybrid_ResNet_LSTM'] = 0
        
        print("\n5. Training Autoencoder + Classifier...")
        try:
            autoencoder, ae_classifier = self.model_factory.create_autoencoder_classifier(
                input_shape, num_classes
            )
            
            autoencoder.compile(optimizer=optimizers.AdamW(learning_rate=0.001), loss='mse')
            autoencoder.fit(X_train, X_train, epochs=30, batch_size=32, verbose=0)
            
            ae_classifier = self.compile_model(ae_classifier)
            
            history_ae = ae_classifier.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=60,
                batch_size=32,
                callbacks=self.get_callbacks(),
                verbose=0
            )
            
            _, ae_accuracy = ae_classifier.evaluate(X_test, y_test_cat, verbose=0)
            results['Autoencoder_Classifier'] = ae_accuracy
            self.models['Autoencoder_Classifier'] = ae_classifier
            self.history['Autoencoder_Classifier'] = history_ae
            print(f"Autoencoder + Classifier: {ae_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in Autoencoder: {e}")
            results['Autoencoder_Classifier'] = 0
        
        print("\n6. Training PyTorch CNN...")
        try:
            pytorch_accuracy = self.model_factory.train_pytorch_cnn(
                X_train, X_test, y_train, y_test, num_classes
            )
            results['PyTorch_CNN'] = pytorch_accuracy
            print(f"PyTorch CNN: {pytorch_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in PyTorch CNN: {e}")
            results['PyTorch_CNN'] = 0
        
        print("\n7. Creating Advanced Ensemble...")
        try:
            valid_models = [(name, model) for name, model in self.models.items() 
                           if results.get(name, 0) > 0.3]
            
            if len(valid_models) >= 3:
                predictions = []
                weights = []
                
                for name, model in valid_models:
                    pred = model.predict(X_test, verbose=0)
                    predictions.append(pred)
                    weights.append(results[name])
                
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
                
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred_classes)
                results['Advanced_Ensemble'] = ensemble_accuracy
                print(f"Advanced Ensemble: {ensemble_accuracy:.4f}")
                
            else:
                print("Not enough valid models for ensemble")
                results['Advanced_Ensemble'] = 0
                
        except Exception as e:
            print(f"Error in Ensemble: {e}")
            results['Advanced_Ensemble'] = 0
        
        return results
    
    def train_advanced_models(self, X, y):
        print("Training advanced models...")
        
        results = {}
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        num_classes = len(np.unique(y))
        input_shape = X_train.shape[1]
        
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        print("\n1. Training Conformer...")
        try:
            conformer_model = self.model_factory.create_conformer(input_shape, num_classes)
            conformer_model = self.compile_model(conformer_model, learning_rate=0.0005)
            
            history_conformer = conformer_model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=80,
                batch_size=16,
                callbacks=self.get_callbacks(),
                verbose=0
            )
            
            _, conformer_accuracy = conformer_model.evaluate(X_test, y_test_cat, verbose=0)
            results['Conformer_Advanced'] = conformer_accuracy
            self.models['Conformer_Advanced'] = conformer_model
            print(f"Conformer: {conformer_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in Conformer: {e}")
            results['Conformer_Advanced'] = 0
        
        print("\n2. Training WaveNet...")
        try:
            wavenet_model = self.model_factory.create_wavenet(input_shape, num_classes)
            wavenet_model = self.compile_model(wavenet_model)
            
            history_wavenet = wavenet_model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=100,
                batch_size=16,
                callbacks=self.get_callbacks(),
                verbose=0
            )
            
            _, wavenet_accuracy = wavenet_model.evaluate(X_test, y_test_cat, verbose=0)
            results['WaveNet_Advanced'] = wavenet_accuracy
            self.models['WaveNet_Advanced'] = wavenet_model
            print(f"WaveNet: {wavenet_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in WaveNet: {e}")
            results['WaveNet_Advanced'] = 0
        
        print("\n3. Training Capsule Network...")
        try:
            capsule_model = self.model_factory.create_capsule_network(input_shape, num_classes)
            capsule_model = self.compile_model(capsule_model, learning_rate=0.0005)
            
            history_capsule = capsule_model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=60,
                batch_size=16,
                callbacks=self.get_callbacks(),
                verbose=0
            )
            
            _, capsule_accuracy = capsule_model.evaluate(X_test, y_test_cat, verbose=0)
            results['CapsuleNet_Advanced'] = capsule_accuracy
            self.models['CapsuleNet_Advanced'] = capsule_model
            print(f"Capsule Network: {capsule_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in Capsule Network: {e}")
            results['CapsuleNet_Advanced'] = 0
        
        return results