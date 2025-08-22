import numpy as np
from scipy.stats import skew, kurtosis
from config import SPEAKER_PROFILES

def get_speaker_profile(filename):
    try:
        file_num = int(filename.split('.')[0])
        for num_range, profile in SPEAKER_PROFILES.items():
            if file_num in num_range:
                return profile
    except:
        pass
    return {
        "name": "Unknown", "quality": "medium", "clarity": 0.50, 
        "min_samples": 20, "preferred_models": ["RandomForest"]
    }

def ultra_safe_stats(data, prefix):
    features = {}
    
    try:
        if isinstance(data, list):
            if len(data) == 0:
                data = [0.0]
            data = np.array(data, dtype=float)
        elif isinstance(data, (int, float)):
            data = np.array([float(data)])
        else:
            data = np.asarray(data, dtype=float).flatten()
        
        data = data[~np.isnan(data)]
        data = data[~np.isinf(data)]
        
        if len(data) == 0:
            data = np.array([0.0])
        
        features[f'{prefix}_mean'] = float(np.mean(data))
        features[f'{prefix}_std'] = float(np.std(data))
        features[f'{prefix}_max'] = float(np.max(data))
        features[f'{prefix}_min'] = float(np.min(data))
        features[f'{prefix}_range'] = features[f'{prefix}_max'] - features[f'{prefix}_min']
        features[f'{prefix}_median'] = float(np.median(data))
        
        if len(data) > 1:
            try:
                features[f'{prefix}_q25'] = float(np.percentile(data, 25))
                features[f'{prefix}_q75'] = float(np.percentile(data, 75))
                features[f'{prefix}_iqr'] = features[f'{prefix}_q75'] - features[f'{prefix}_q25']
                
                if np.std(data) > 1e-10:
                    features[f'{prefix}_skew'] = float(skew(data))
                    features[f'{prefix}_kurtosis'] = float(kurtosis(data))
                else:
                    features[f'{prefix}_skew'] = 0.0
                    features[f'{prefix}_kurtosis'] = 0.0
                    
            except:
                features[f'{prefix}_q25'] = features[f'{prefix}_min']
                features[f'{prefix}_q75'] = features[f'{prefix}_max']
                features[f'{prefix}_iqr'] = features[f'{prefix}_range']
                features[f'{prefix}_skew'] = 0.0
                features[f'{prefix}_kurtosis'] = 0.0
        else:
            features[f'{prefix}_q25'] = features[f'{prefix}_mean']
            features[f'{prefix}_q75'] = features[f'{prefix}_mean']
            features[f'{prefix}_iqr'] = 0.0
            features[f'{prefix}_skew'] = 0.0
            features[f'{prefix}_kurtosis'] = 0.0
        
        try:
            features[f'{prefix}_energy'] = float(np.sum(data**2))
            features[f'{prefix}_rms'] = float(np.sqrt(np.mean(data**2)))
        except:
            features[f'{prefix}_energy'] = 0.0
            features[f'{prefix}_rms'] = 0.0
        
        if abs(features[f'{prefix}_mean']) > 1e-10:
            features[f'{prefix}_cv'] = features[f'{prefix}_std'] / abs(features[f'{prefix}_mean'])
        else:
            features[f'{prefix}_cv'] = 0.0
            
    except Exception as e:
        print(f"Error calculating stats for {prefix}: {e}")
        default_stats = ['mean', 'std', 'max', 'min', 'range', 'median', 'q25', 'q75', 'iqr', 'skew', 'kurtosis', 'energy', 'rms', 'cv']
        for stat in default_stats:
            features[f'{prefix}_{stat}'] = 0.0
    
    clean_features = {}
    for key, value in features.items():
        if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
            clean_features[key] = float(value)
        else:
            clean_features[key] = 0.0
    
    return clean_features
