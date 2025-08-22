import os
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from data_loader import DataLoader
from data_balancer import DataBalancer
from model_trainer import ModelTrainer


class ASRSystem:
    def __init__(self, data_path="C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean", output_path="output_files"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.data_loader = DataLoader(data_path)
        self.data_balancer = DataBalancer()
        self.model_trainer = ModelTrainer()
        
        self.word_frequencies = {}
        self.scaler = None
        self.label_encoder = None
        
    def run_ultimate_evaluation(self):
        print("Starting ASR Evaluation...")
        print("="*60)
        
        df, word_counts, speaker_word_counts = self.data_loader.load_maximum_data()
        
        if df is None or len(df) == 0:
            print("FAILED: No data loaded!")
            return None
        
        feature_cols = [col for col in df.columns if col not in ['file_path', 'word', 'speaker']]
        X = df[feature_cols].values
        y = df['word'].values
        speakers = df['speaker'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        scalers = {
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler()
        }
        
        best_scaler = None
        best_scaler_score = 0
        
        print(f"Testing scaling methods...")
        for scaler_name, scaler in scalers.items():
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            score = rf.score(X_test, y_test)
            
            print(f"  {scaler_name}: {score:.4f}")
            
            if score > best_scaler_score:
                best_scaler_score = score
                best_scaler = scaler
                self.scaler = scaler
        
        X_scaled = best_scaler.fit_transform(X)
        
        print(f"\nDataset Info:")
        print(f"  Samples: {len(df)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Words: {len(np.unique(y))}")
        print(f"  Speakers: {len(np.unique(speakers))}")
        print(f"  Best scaler: {type(best_scaler).__name__}")
        
        X_balanced, y_balanced, speakers_balanced = self.data_balancer.apply_smart_data_balancing(
            X_scaled, y_encoded, speakers
        )
        
        model_results = self.model_trainer.train_ultimate_models(X_balanced, y_balanced, speakers_balanced)
        
        best_model_name = max(model_results, key=model_results.get)
        best_accuracy = model_results[best_model_name]
        
        final_results = {
            'dataset_info': {
                'original_samples': len(df),
                'balanced_samples': len(X_balanced),
                'features': len(feature_cols),
                'words': len(np.unique(y)),
                'speakers': len(np.unique(speakers)),
                'word_frequencies': dict(word_counts)
            },
            'model_results': model_results,
            'absolute_best': {
                'model': best_model_name,
                'accuracy': best_accuracy
            },
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_path / 'ultimate_high_accuracy_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        models_to_save = {
            'models': self.model_trainer.models,
            'scaler': best_scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': feature_cols
        }
        
        pickle.dump(models_to_save, open(self.output_path / 'ultimate_models.pkl', 'wb'))
        
        print(f"\n" + "="*60)
        print(f"ASR EVALUATION RESULTS")
        print(f"="*60)
        print(f"Dataset: {len(df)} → {len(X_balanced)} samples, {len(feature_cols)} features")
        print(f"Words: {len(np.unique(y))}")
        print(f"")
        print(f"BEST RESULT:")
        print(f"   Model: {best_model_name}")
        print(f"   Accuracy: {best_accuracy*100:.2f}%")
        print(f"")
        print(f"ALL RESULTS:")
        for model, acc in sorted(model_results.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model}: {acc*100:.2f}%")
        print(f"="*60)
        
        return final_results