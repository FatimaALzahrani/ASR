import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

class DataProcessor:
    
    def __init__(self):
        self.speaker_scalers = {}
    
    def prepare_data(self, X, y, speaker_name):
        print(f"Processing data for speaker: {speaker_name}")
        
        try:
            print(f"Handling missing values...")
            imputer = SimpleImputer(strategy='constant', fill_value=0.0)
            X_imputed = imputer.fit_transform(X)
            
            X_clean = np.nan_to_num(X_imputed, nan=0.0, posinf=1e6, neginf=-1e6)
            
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            n_samples = X_scaled.shape[0]
            n_features = X_scaled.shape[1]
            
            max_features_by_samples = min(100, n_samples // 3)
            max_features_by_features = min(150, n_features // 2)
            target_features = max(10, min(max_features_by_samples, max_features_by_features))
            
            print(f"Selecting {target_features} features from {n_features} original features")
            
            try:
                if target_features < n_features:
                    selector = SelectKBest(score_func=f_classif, k=target_features)
                    X_selected = selector.fit_transform(X_scaled, y)
                else:
                    X_selected = X_scaled
                    selector = None
            except Exception:
                X_selected = X_scaled[:, :target_features] if n_features > target_features else X_scaled
                selector = None
            
            if X_selected.shape[1] > 80 and X_selected.shape[0] > 20:
                try:
                    max_components_samples = X_selected.shape[0] - 1
                    max_components_features = X_selected.shape[1] - 1
                    target_components = min(80, max_components_samples, max_components_features)
                    target_components = max(10, target_components)
                    
                    if target_components < X_selected.shape[1]:
                        pca = PCA(n_components=target_components, random_state=42)
                        X_final = pca.fit_transform(X_selected)
                        print(f"Applied PCA: {X_selected.shape[1]} → {X_final.shape[1]}")
                    else:
                        X_final = X_selected
                        pca = None
                except Exception:
                    X_final = X_selected
                    pca = None
            else:
                X_final = X_selected
                pca = None
            
            self.speaker_scalers[speaker_name] = {
                'imputer': imputer,
                'scaler': scaler,
                'selector': selector,
                'pca': pca
            }
            
            print(f"Result: {X.shape[1]} → {X_final.shape[1]} features")
            return X_final
            
        except Exception as e:
            print(f"Data preparation failed: {e}")
            try:
                scaler = RobustScaler()
                X_simple = scaler.fit_transform(np.nan_to_num(X, nan=0.0))
                self.speaker_scalers[speaker_name] = {'scaler': scaler}
                return X_simple
            except:
                return np.nan_to_num(X, nan=0.0)
    
    def encode_labels(self, y):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        return y_encoded, label_encoder
    
    def get_scaler(self, speaker_name):
        return self.speaker_scalers.get(speaker_name, None)
    
    def save_scaler(self, speaker_name, label_encoder):
        if speaker_name in self.speaker_scalers:
            self.speaker_scalers[speaker_name]['label_encoder'] = label_encoder
