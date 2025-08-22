import numpy as np

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("imbalanced-learn not available, using manual balancing")


class DataBalancer:
    def __init__(self):
        self.noise_level = 0.005
        self.target_ratio = 0.3
    
    def balance_data(self, X, y, speakers):
        print("Applying data balancing...")
        
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"Original distribution: {len(unique_classes)} classes")
        print(f"  Min: {np.min(class_counts)}, Max: {np.max(class_counts)}, Mean: {np.mean(class_counts):.1f}")
        
        if IMBLEARN_AVAILABLE:
            try:
                min_samples = np.min(class_counts)
                k_neighbors = min(3, min_samples - 1) if min_samples > 1 else 1
                
                if k_neighbors >= 1 and min_samples >= 2:
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_balanced, y_balanced = smote.fit_resample(X, y)
                    
                    speakers_balanced = np.array(speakers.tolist() * (len(X_balanced) // len(speakers) + 1))[:len(X_balanced)]
                    
                    print(f"SMOTE applied: {len(X)} -> {len(X_balanced)} samples")
                    return X_balanced, y_balanced, speakers_balanced
                else:
                    print("SMOTE not applicable, using manual balancing")
            except Exception as e:
                print(f"SMOTE failed: {e}")
        
        print("Applying manual augmentation...")
        
        X_list = X.tolist()
        y_list = y.tolist()
        speakers_list = speakers.tolist()
        
        target_min = max(3, int(np.mean(class_counts) * self.target_ratio))
        
        for class_label in unique_classes:
            class_mask = y == class_label
            class_count = np.sum(class_mask)
            
            if class_count < target_min:
                class_indices = np.where(class_mask)[0]
                duplications_needed = target_min - class_count
                
                for _ in range(duplications_needed):
                    idx = np.random.choice(class_indices)
                    
                    noise = np.random.normal(0, self.noise_level, X[idx].shape)
                    augmented_sample = X[idx] + noise
                    
                    X_list.append(augmented_sample)
                    y_list.append(y[idx])
                    speakers_list.append(speakers[idx])
        
        X_balanced = np.array(X_list)
        y_balanced = np.array(y_list)
        speakers_balanced = np.array(speakers_list)
        
        print(f"Manual balancing: {len(X)} -> {len(X_balanced)} samples")
        return X_balanced, y_balanced, speakers_balanced