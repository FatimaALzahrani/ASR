import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from fixed_audio_dataset import FixedAudioDataset
import torch


class BaselineEvaluator:
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.train_df = trainer.train_df
        self.test_df = trainer.test_df
        self.label_encoder = trainer.label_encoder
    
    def run_baseline_comparison(self):
        print("Running baseline model comparison...")
        
        print("Preparing k-NN data...")
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []
        
        for idx, row in self.train_df.iterrows():
            dataset = FixedAudioDataset(pd.DataFrame([row]), feature_type='mfcc', augment=False)
            try:
                features, word, _ = dataset[0]
                if torch.any(features):
                    train_features.append(np.mean(features.numpy(), axis=0))
                    train_labels.append(word)
            except:
                continue
                
            if (idx + 1) % 100 == 0:
                print(f"Processing train: {idx + 1}/{len(self.train_df)}")
        
        for idx, row in self.test_df.iterrows():
            dataset = FixedAudioDataset(pd.DataFrame([row]), feature_type='mfcc', augment=False)
            try:
                features, word, _ = dataset[0]
                if torch.any(features):
                    test_features.append(np.mean(features.numpy(), axis=0))
                    test_labels.append(word)
            except:
                continue
                
            if (idx + 1) % 50 == 0:
                print(f"Processing test: {idx + 1}/{len(self.test_df)}")
        
        if not train_features or not test_features:
            print("No valid features for baseline comparison")
            return 0.0
        
        X_train = np.array(train_features)
        X_test = np.array(test_features)
        
        y_train = self.label_encoder.transform(train_labels)
        y_test = self.label_encoder.transform(test_labels)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print("Training k-NN...")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)
        knn_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"k-NN accuracy: {knn_accuracy:.4f}")
        
        print(f"\nk-NN performance by speaker:")
        for i, (true_label, pred_label) in enumerate(zip(test_labels, self.label_encoder.inverse_transform(y_pred))):
            speaker = self.test_df.iloc[i]['speaker'] if i < len(self.test_df) else 'unknown'
        
        return knn_accuracy