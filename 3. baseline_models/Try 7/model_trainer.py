from config import *
from data_loader import DataLoader

class ModelTrainer:
    def __init__(self, random_state=Config.RANDOM_STATE):
        self.random_state = random_state
        self.data_loader = DataLoader(random_state)
        
    def get_model_definitions(self, model_type='speaker'):
        if model_type == 'speaker':
            return {
                'Random_Forest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    random_state=self.random_state, n_jobs=-1
                ),
                'Extra_Trees': ExtraTreesClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    random_state=self.random_state, n_jobs=-1
                ),
                'Gradient_Boosting': GradientBoostingClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state
                ),
                'SVM_RBF': SVC(
                    C=1.0, gamma='scale', random_state=self.random_state
                ),
                'Logistic_Regression': LogisticRegression(
                    max_iter=1000, random_state=self.random_state
                )
            }
        else:
            return {
                'Global_Random_Forest': RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    random_state=self.random_state, n_jobs=-1
                ),
                'Global_Extra_Trees': ExtraTreesClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    random_state=self.random_state, n_jobs=-1
                ),
                'Global_Gradient_Boosting': GradientBoostingClassifier(
                    n_estimators=150, max_depth=8, learning_rate=0.1,
                    random_state=self.random_state
                ),
                'Global_SVM': SVC(
                    C=1.0, gamma='scale', random_state=self.random_state
                ),
                'Global_Logistic_Regression': LogisticRegression(
                    max_iter=1000, random_state=self.random_state
                )
            }
    
    def train_speaker_specific_models(self, processed_data):
        print("Training speaker-specific models...")
        
        if processed_data is None:
            raise ValueError("No processed data available.")
        
        train_df, test_df = self.data_loader.create_robust_splits(processed_data)
        
        speaker_results = {}
        
        for speaker in train_df['speaker'].unique():
            print(f"Training model for speaker: {speaker}")
            
            speaker_train = train_df[train_df['speaker'] == speaker]
            speaker_test = test_df[test_df['speaker'] == speaker]
            
            if len(speaker_train) < 5 or len(speaker_test) < 2:
                print(f"Insufficient data for {speaker}, skipping...")
                continue
            
            if speaker_train['word'].nunique() < 3:
                print(f"Too few words for {speaker}, skipping...")
                continue
            
            X_train, y_train, _ = self.data_loader.prepare_consistent_features(speaker_train)
            X_test, y_test, _ = self.data_loader.prepare_consistent_features(speaker_test)
            
            common_features = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_features]
            X_test = X_test[common_features]
            
            if len(common_features) < 10:
                print(f"Too few features for {speaker}, skipping...")
                continue
            
            n_features_to_select = min(Config.FEATURE_SELECTION_K, len(common_features))
            selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
            
            try:
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
            except Exception as e:
                print(f"Feature selection failed for {speaker}: {str(e)}")
                X_train_selected = X_train.values
                X_test_selected = X_test.values
                selector = None
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            models = self.get_model_definitions('speaker')
            
            speaker_model_results = {}
            
            for name, model in models.items():
                try:
                    print(f"  Training {name} for {speaker}...")
                    
                    model.fit(X_train_scaled, y_train)
                    
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    speaker_model_results[name] = {
                        'model': model,
                        'scaler': scaler,
                        'selector': selector,
                        'accuracy': accuracy,
                        'predictions': y_pred,
                        'true_labels': y_test.values,
                        'feature_columns': common_features
                    }
                    
                    print(f"    {name} accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    
                except Exception as e:
                    print(f"    Error training {name}: {str(e)}")
            
            speaker_results[speaker] = speaker_model_results
        
        return speaker_results
    
    def train_global_models(self, processed_data):
        print("Training global models...")
        
        if processed_data is None:
            raise ValueError("No processed data available.")
        
        train_df, test_df = self.data_loader.create_robust_splits(processed_data)
        
        X_train, y_train, train_features = self.data_loader.prepare_consistent_features(train_df)
        X_test, y_test, test_features = self.data_loader.prepare_consistent_features(test_df)
        
        common_features = list(set(train_features) & set(test_features))
        X_train = X_train[common_features]
        X_test = X_test[common_features]
        
        n_features_to_select = min(Config.GLOBAL_FEATURE_SELECTION_K, X_train.shape[1])
        selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        models = self.get_model_definitions('global')
        
        global_results = {}
        
        for name, model in models.items():
            try:
                print(f"Training {name}...")
                
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                global_results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'selector': selector,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'true_labels': y_test.values,
                    'test_speakers': test_df['speaker'].values
                }
                
                print(f"{name} accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
        
        return global_results