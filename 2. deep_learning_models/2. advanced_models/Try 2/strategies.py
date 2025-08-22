import numpy as np
from sklearn.model_selection import train_test_split, KFold


class TrainingStrategies:
    
    def __init__(self, all_data, trainer):
        self.all_data = all_data
        self.trainer = trainer
    
    def strategy_1_simple_split(self):
        print("\n" + "="*60)
        print("Strategy 1: Simple 90/10 Split")
        print("="*60)
        
        train_data, test_data = train_test_split(
            self.all_data, test_size=0.1, random_state=42, 
            stratify=self.all_data['word']
        )
        
        print(f"New split:")
        print(f"  Training: {len(train_data)} samples ({len(train_data)/len(self.all_data)*100:.1f}%)")
        print(f"  Testing: {len(test_data)} samples ({len(test_data)/len(self.all_data)*100:.1f}%)")
        
        print(f"\nTraining HMM-DNN with 90% of data...")
        accuracy = self.trainer.train_single_model(train_data, test_data, "Strategy1_90_10")
        
        return accuracy
    
    def strategy_2_cross_validation(self):
        print("\n" + "="*60)
        print("Strategy 2: 5-Fold Cross Validation")
        print("="*60)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        accuracies = []
        fold = 1
        
        for train_idx, test_idx in kfold.split(self.all_data):
            print(f"\nFold {fold}/5:")
            
            train_data = self.all_data.iloc[train_idx]
            test_data = self.all_data.iloc[test_idx]
            
            print(f"  Training: {len(train_data)} samples")
            print(f"  Testing: {len(test_data)} samples")
            
            accuracy = self.trainer.train_single_model(train_data, test_data, f"Strategy2_Fold{fold}")
            accuracies.append(accuracy)
            
            print(f"  Fold {fold} accuracy: {accuracy:.4f}")
            fold += 1
        
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f"\nCross Validation results:")
        print(f"  Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"  Best fold: {max(accuracies):.4f}")
        print(f"  Worst fold: {min(accuracies):.4f}")
        
        return mean_accuracy, std_accuracy
    
    def strategy_3_leave_one_speaker_out(self):
        print("\n" + "="*60)
        print("Strategy 3: Leave-One-Speaker-Out (LOSO)")
        print("="*60)
        
        speakers = self.all_data['speaker'].unique()
        speaker_accuracies = {}
        
        for test_speaker in speakers:
            print(f"\nTesting on: {test_speaker}")
            
            train_data = self.all_data[self.all_data['speaker'] != test_speaker]
            test_data = self.all_data[self.all_data['speaker'] == test_speaker]
            
            print(f"  Training: {len(train_data)} samples (4 speakers)")
            print(f"  Testing: {len(test_data)} samples ({test_speaker})")
            
            accuracy = self.trainer.train_single_model(train_data, test_data, f"Strategy3_LOSO_{test_speaker}")
            speaker_accuracies[test_speaker] = accuracy
            
            print(f"  {test_speaker} accuracy: {accuracy:.4f}")
        
        mean_accuracy = np.mean(list(speaker_accuracies.values()))
        
        print(f"\nLOSO results:")
        print(f"  Mean accuracy: {mean_accuracy:.4f}")
        for speaker, acc in sorted(speaker_accuracies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {speaker}: {acc:.4f}")
        
        return mean_accuracy, speaker_accuracies
    
    def strategy_4_full_data_with_regularization(self):
        print("\n" + "="*60)
        print("Strategy 4: Full Data + Strong Regularization")
        print("="*60)
        
        print(f"Using all data for training: {len(self.all_data)} samples")
        print(f"With strong regularization to prevent overfitting")
        
        accuracy = self.trainer.train_full_data_model()
        
        return accuracy