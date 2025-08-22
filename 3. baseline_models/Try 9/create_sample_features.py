import os
import pandas as pd
import numpy as np
from file_utils import ensure_directory_exists, save_dataframe
from settings import FEATURES_PATH


def create_sample_features():
    print("Creating sample features for testing...")
    
    # Sample speakers and words (Arabic names and words)
    speakers = ["أحمد", "عاصم", "هيفاء", "أسيل", "وسام"]
    words = ["ماما", "بابا", "ماء", "خبز", "حليب", "كتاب", "قلم", "شمس", "قمر", "بيت"]
    
    # Generate sample data
    sample_data = []
    
    np.random.seed(42)  # For reproducible results
    
    for speaker in speakers:
        for word in words:
            # Create multiple samples per speaker-word combination
            for sample_num in range(np.random.randint(3, 8)):  # 3-7 samples per word
                
                # Create realistic-looking feature values
                features = {}
                
                # File info
                features['file_path'] = f"sample_data/{speaker}/{word}_{sample_num:02d}.wav"
                features['word'] = word
                features['speaker'] = speaker
                features['actual_duration'] = np.random.uniform(0.5, 2.5)
                
                # Basic audio features
                features['rms_energy'] = np.random.uniform(0.01, 0.3)
                features['energy_mean'] = np.random.uniform(0.001, 0.1)
                features['energy_std'] = np.random.uniform(0.001, 0.05)
                features['zcr_mean'] = np.random.uniform(0.02, 0.15)
                features['zcr_std'] = np.random.uniform(0.01, 0.08)
                
                # MFCC features (13 coefficients with statistics)
                for i in range(13):
                    base_mfcc = np.random.normal(0, 10)
                    features[f'mfcc_{i}_mean'] = base_mfcc + np.random.normal(0, 2)
                    features[f'mfcc_{i}_std'] = np.random.uniform(0.5, 5.0)
                    features[f'mfcc_{i}_min'] = features[f'mfcc_{i}_mean'] - np.random.uniform(5, 15)
                    features[f'mfcc_{i}_max'] = features[f'mfcc_{i}_mean'] + np.random.uniform(5, 15)
                    
                    # Delta MFCC
                    features[f'delta_mfcc_{i}_mean'] = np.random.normal(0, 1)
                    features[f'delta_mfcc_{i}_std'] = np.random.uniform(0.2, 2.0)
                    
                    # Delta-Delta MFCC
                    features[f'delta2_mfcc_{i}_mean'] = np.random.normal(0, 0.5)
                    features[f'delta2_mfcc_{i}_std'] = np.random.uniform(0.1, 1.0)
                
                # Spectral features
                features['spectral_centroid_mean'] = np.random.uniform(1000, 4000)
                features['spectral_centroid_std'] = np.random.uniform(200, 800)
                features['spectral_bandwidth_mean'] = np.random.uniform(800, 2000)
                features['spectral_bandwidth_std'] = np.random.uniform(100, 400)
                features['spectral_contrast_mean'] = np.random.uniform(10, 30)
                features['spectral_contrast_std'] = np.random.uniform(2, 8)
                features['spectral_rolloff_mean'] = np.random.uniform(2000, 6000)
                features['spectral_rolloff_std'] = np.random.uniform(300, 1000)
                
                # Mel spectrogram features
                for i in range(10):
                    features[f'mel_{i}_mean'] = np.random.uniform(-20, 10)
                    features[f'mel_{i}_std'] = np.random.uniform(1, 8)
                
                # F0 (pitch) features
                f0_mean = np.random.uniform(100, 300)  # Typical for children
                features['f0_mean'] = f0_mean
                features['f0_std'] = np.random.uniform(10, 50)
                features['f0_min'] = f0_mean - np.random.uniform(20, 80)
                features['f0_max'] = f0_mean + np.random.uniform(20, 80)
                features['f0_range'] = features['f0_max'] - features['f0_min']
                features['f0_skew'] = np.random.uniform(-1, 1)
                features['f0_kurtosis'] = np.random.uniform(-2, 2)
                
                # Additional energy features
                features['energy_skew'] = np.random.uniform(-2, 2)
                features['energy_kurtosis'] = np.random.uniform(-2, 4)
                features['tempo'] = np.random.uniform(60, 120)
                features['silence_ratio'] = np.random.uniform(0.1, 0.4)
                features['speech_ratio'] = 1.0 - features['silence_ratio']
                features['snr_estimate'] = np.random.uniform(10, 40)
                
                # Composite features
                features['spectral_ratio'] = features['spectral_centroid_mean'] / (features['spectral_bandwidth_mean'] + 1e-8)
                features['mfcc_ratio_0_1'] = features['mfcc_0_mean'] / (features['mfcc_1_mean'] + 1e-8)
                features['energy_ratio'] = features['rms_energy'] / (features['energy_mean'] + 1e-8)
                features['f0_cv'] = features['f0_std'] / (features['f0_mean'] + 1e-8)
                
                sample_data.append(features)
    
    # Create DataFrame
    features_df = pd.DataFrame(sample_data)
    
    print(f"Created {len(features_df)} sample records")
    print(f"Features: {len(features_df.columns)} columns")
    print(f"Speakers: {features_df['speaker'].nunique()} unique")
    print(f"Words: {features_df['word'].nunique()} unique")
    
    # Display distribution
    print("\nSpeaker distribution:")
    print(features_df['speaker'].value_counts())
    
    print("\nWord distribution:")
    print(features_df['word'].value_counts())
    
    return features_df


def save_sample_features(features_df):
    ensure_directory_exists(FEATURES_PATH)
    
    output_file = os.path.join(FEATURES_PATH, "features_for_modeling.csv")
    
    success = save_dataframe(features_df, output_file, format='csv')
    
    if success:
        print(f"\nSample features saved to: {output_file}")
        
        # Create info file
        info = {
            "description": "Sample features for testing the speech recognition system",
            "total_samples": len(features_df),
            "speakers": list(features_df['speaker'].unique()),
            "words": list(features_df['word'].unique()),
            "features_count": len(features_df.columns) - 4,
            "note": "This is synthetic data for testing purposes only"
        }
        
        import json
        info_file = os.path.join(FEATURES_PATH, "sample_data_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        print(f"Data info saved to: {info_file}")
        print("\nYou can now run: python main.py train")
        return True
    else:
        print("Failed to save sample features")
        return False


def main():
    print("Sample Feature Generator")
    print("=" * 50)
    print("Creating synthetic features for testing the speech recognition system")
    print("=" * 50)
    
    features_df = create_sample_features()
    success = save_sample_features(features_df)
    
    if success:
        print("\n✅ Sample data created successfully!")
        print("\nNext steps:")
        print("1. Run: python main.py train")
        print("2. Run: python main.py test")
        print("\nNote: This is synthetic data. For real usage, collect actual audio files")
        print("and use extract_features.py to create features from real audio.")
        return 0
    else:
        print("\n❌ Failed to create sample data")
        return 1


if __name__ == "__main__":
    exit(main())
