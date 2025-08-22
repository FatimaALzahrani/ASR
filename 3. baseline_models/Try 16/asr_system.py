import os
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import RobustScaler, LabelEncoder
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from utils import print_results_summary

class ASRSystem:
    def __init__(self, data_path="C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Codes/Real Codes/01_data_processing2/data/clean", output_path="asr_results"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.data_processor = DataProcessor(self.data_path)
        self.model_trainer = ModelTrainer()
        
        self.scaler = None
        self.label_encoder = None
    
    def run_evaluation(self):
        print("Starting ASR System Evaluation")
        print("="*50)
        
        df = self.data_processor.load_and_prepare_data()
        
        if df is None or len(df) == 0:
            print("No data loaded!")
            return None
        
        balanced_df = self.data_processor.apply_balancing(df)
        
        feature_cols = [col for col in balanced_df.columns if col not in ['file_path', 'word', 'speaker']]
        X = balanced_df[feature_cols].values
        y = balanced_df['word'].values
        speakers = balanced_df['speaker'].values
        
        print(f"\nDataset Information:")
        print(f"Samples: {len(balanced_df)}")
        print(f"Features: {len(feature_cols)}")
        print(f"Words: {len(np.unique(y))}")
        print(f"Speakers: {len(np.unique(speakers))}")
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        general_results, X_test, y_test = self.model_trainer.train_general_models(X_scaled, y_encoded)
        
        speaker_results = self.model_trainer.train_speaker_models(X_scaled, y_encoded, speakers)
        
        final_results = {
            'dataset_info': {
                'samples': len(balanced_df),
                'features': len(feature_cols),
                'words': len(np.unique(y)),
                'speakers': len(np.unique(speakers))
            },
            'general_models': general_results,
            'speaker_models': speaker_results,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_path / 'asr_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        general_models, speaker_models = self.model_trainer.get_models()
        models_to_save = {
            'general_models': general_models,
            'speaker_models': speaker_models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': feature_cols
        }
        
        pickle.dump(models_to_save, open(self.output_path / 'asr_models.pkl', 'wb'))
        
        print(f"\n" + "="*50)
        print(f"ASR SYSTEM RESULTS")
        print(f"="*50)
        
        print_results_summary(final_results)
        
        print(f"\nEvaluation complete!")
        print(f"Results saved to: {self.output_path}/")
        print(f"="*50)
        
        return final_results
