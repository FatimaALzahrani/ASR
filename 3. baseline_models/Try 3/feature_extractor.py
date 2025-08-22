import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler


class FeatureExtractor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        
    def extract_features(self, data):
        features_list = []
        labels_list = []
        
        for idx, row in data.iterrows():
            try:
                features = [
                    row.get('duration', 0),
                    row.get('rms_energy', 0),
                    row.get('snr', 0),
                    row.get('quality_score', 0)
                ]
                
                word = str(row['word']) if pd.notna(row['word']) else ''
                
                features.extend([
                    len(word),
                    word.count('ا'),
                    word.count('ي'),
                    word.count('و'),
                    1 if any(c in word for c in 'ضصثقفغعهخحجد') else 0,
                ])
                
                speaker = row.get('speaker', '')
                speaker_features = [
                    1 if speaker == 'أحمد' else 0,
                    1 if speaker == 'أسيل' else 0,
                    1 if speaker == 'هيفاء' else 0,
                    1 if speaker == 'وسام' else 0,
                    1 if speaker == 'عاصم' else 0,
                ]
                features.extend(speaker_features)
                
                features = [f if not (np.isnan(f) or np.isinf(f)) else 0 for f in features]
                
                features_list.append(features)
                labels_list.append(row['word'])
                
            except Exception as e:
                continue
        
        return np.array(features_list), np.array(labels_list)
    
    def preprocess_data(self, train_data, test_data):
        X_train, y_train = self.extract_features(train_data)
        X_test, y_test = self.extract_features(test_data)
        
        train_words = set(y_train)
        test_words = set(y_test)
        common_words = train_words.intersection(test_words)
        
        train_mask = np.isin(y_train, list(common_words))
        test_mask = np.isin(y_test, list(common_words))
        
        X_train_filtered = X_train[train_mask]
        y_train_filtered = y_train[train_mask]
        X_test_filtered = X_test[test_mask]
        y_test_filtered = y_test[test_mask]
        
        self.label_encoder.fit(list(common_words))
        
        y_train_encoded = self.label_encoder.transform(y_train_filtered)
        y_test_encoded = self.label_encoder.transform(y_test_filtered)
        
        X_train_scaled = self.scaler.fit_transform(X_train_filtered)
        X_test_scaled = self.scaler.transform(X_test_filtered)
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, common_words