import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.selected_features = None
    
    def load_features(self, features_path):
        features_file = os.path.join(features_path, "features_for_modeling.csv")
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        df = pd.read_csv(features_file)
        print(f"Loaded {len(df)} samples")
        return df
    
    def prepare_features(self, df, target_column='word'):
        feature_columns = [col for col in df.columns 
                          if col not in ['file_path', 'word', 'speaker', 'quality', 'actual_duration']]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        X = X.fillna(X.mean())
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y.astype(str))
        
        unique_labels, counts = np.unique(y_encoded, return_counts=True)
        valid_labels = unique_labels[counts >= 2]
        
        if len(valid_labels) < 2:
            raise ValueError("Insufficient valid classes for training")
        
        valid_mask = np.isin(y_encoded, valid_labels)
        X = X[valid_mask]
        y_encoded = y_encoded[valid_mask]
        
        print(f"Filtered data: {len(X)} samples, {len(valid_labels)} classes")
        return X, y_encoded, valid_mask
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def select_features(self, X, y, k=80):
        unique_labels, counts = np.unique(y, return_counts=True)
        valid_labels = unique_labels[counts >= 2]
        valid_mask = np.isin(y, valid_labels)
        
        X_filtered = X[valid_mask]
        y_filtered = y[valid_mask]
        
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X_filtered.shape[1]))
        X_selected = self.feature_selector.fit_transform(X_filtered, y_filtered)
        
        if hasattr(X, 'columns'):
            self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        else:
            self.selected_features = list(range(X_selected.shape[1]))
        
        print(f"Selected {len(self.selected_features)} features from {X.shape[1]}")
        return X_selected, y_filtered, valid_mask
    
    def create_composite_features(self, X):
        X_new = X.copy()
        
        if hasattr(X, 'columns'):
            columns = X.columns
            
            if 'spectral_centroid_mean' in columns and 'spectral_bandwidth_mean' in columns:
                X_new['spectral_ratio'] = X['spectral_centroid_mean'] / (X['spectral_bandwidth_mean'] + 1e-8)
            
            mfcc_cols = [col for col in columns if 'mfcc_' in col and '_mean' in col]
            if len(mfcc_cols) >= 2:
                X_new['mfcc_ratio_0_1'] = X[mfcc_cols[0]] / (X[mfcc_cols[1]] + 1e-8)
            
            if 'rms_energy' in columns and 'energy_mean' in columns:
                X_new['energy_ratio'] = X['rms_energy'] / (X['energy_mean'] + 1e-8)
            
            if 'f0_mean' in columns and 'f0_std' in columns:
                X_new['f0_cv'] = X['f0_std'] / (X['f0_mean'] + 1e-8)
        
        print(f"Created {len(X_new.columns) - len(X.columns)} composite features")
        return X_new
    
    def get_speaker_data(self, df, speaker):
        speaker_data = df[df['speaker'] == speaker].copy()
        return speaker_data
    
    def get_data_distribution(self, df):
        word_counts = df['word'].value_counts()
        speaker_counts = df['speaker'].value_counts()
        
        distribution = {
            'total_samples': len(df),
            'unique_words': len(word_counts),
            'unique_speakers': len(speaker_counts),
            'word_distribution': word_counts.to_dict(),
            'speaker_distribution': speaker_counts.to_dict()
        }
        
        return distribution