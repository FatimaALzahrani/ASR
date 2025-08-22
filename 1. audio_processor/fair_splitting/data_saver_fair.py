import json
from pathlib import Path


class DataSaverFair:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
    
    def save_fair_splits(self, train_df, val_df, test_df, old_train_df, old_val_df, old_test_df):
        print(f"Saving fair splits...")
        
        backup_dir = self.data_path / "backup_unfair"
        backup_dir.mkdir(exist_ok=True)
        
        try:
            old_train_df.to_csv(backup_dir / "train.csv", index=False, encoding='utf-8')
            old_val_df.to_csv(backup_dir / "validation.csv", index=False, encoding='utf-8')
            old_test_df.to_csv(backup_dir / "test.csv", index=False, encoding='utf-8')
            print(f"Backup of old split saved: {backup_dir}")
        except:
            print("Warning: Could not create backup")
        
        train_df.to_csv(self.data_path / "train.csv", index=False, encoding='utf-8')
        val_df.to_csv(self.data_path / "validation.csv", index=False, encoding='utf-8')  
        test_df.to_csv(self.data_path / "test.csv", index=False, encoding='utf-8')
        
        print(f"Fair split saved to: {self.data_path}")
        
        self._save_split_statistics(train_df, val_df, test_df)
    
    def _save_split_statistics(self, train_df, val_df, test_df):
        stats = {
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'train_samples': len(train_df),
            'validation_samples': len(val_df),
            'test_samples': len(test_df),
            'speaker_distribution': {
                'train': train_df['speaker'].value_counts().to_dict() if not train_df.empty else {},
                'validation': val_df['speaker'].value_counts().to_dict() if not val_df.empty else {},
                'test': test_df['speaker'].value_counts().to_dict() if not test_df.empty else {}
            },
            'word_distribution': {
                'train': len(train_df['word'].unique()) if not train_df.empty else 0,
                'validation': len(val_df['word'].unique()) if not val_df.empty else 0,
                'test': len(test_df['word'].unique()) if not test_df.empty else 0
            }
        }
        
        with open(self.data_path / 'fair_split_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Split statistics saved: fair_split_stats.json")