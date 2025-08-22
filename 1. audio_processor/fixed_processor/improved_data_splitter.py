import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict


class ImprovedDataSplitter:
    def create_improved_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        print("\nCreating improved data splits...")
        
        word_counts = df['word'].value_counts()
        
        train_data = []
        val_data = []
        test_data = []
        
        for word in df['word'].unique():
            word_data = df[df['word'] == word].copy()
            sample_count = len(word_data)
            
            if sample_count == 1:
                train_data.append(word_data)
                print(f"{word}: 1 sample → training only")
                
            elif sample_count == 2:
                train_data.append(word_data.iloc[:1])
                test_data.append(word_data.iloc[1:])
                print(f"{word}: 2 samples → training (1) + test (1)")
                
            elif sample_count == 3:
                train_data.append(word_data.iloc[:2])
                test_data.append(word_data.iloc[2:])
                print(f"{word}: 3 samples → training (2) + test (1)")
                
            elif sample_count == 4:
                train_data.append(word_data.iloc[:2])
                val_data.append(word_data.iloc[2:3])
                test_data.append(word_data.iloc[3:])
                print(f"{word}: 4 samples → training (2) + validation (1) + test (1)")
                
            else:
                try:
                    if len(word_data['speaker'].unique()) > 1 and sample_count >= 6:
                        train_part, temp_part = train_test_split(
                            word_data, test_size=0.4, random_state=42, 
                            stratify=word_data['speaker']
                        )
                        if len(temp_part) >= 2:
                            val_part, test_part = train_test_split(
                                temp_part, test_size=0.5, random_state=42
                            )
                        else:
                            val_part = temp_part.iloc[:len(temp_part)//2] if len(temp_part) > 1 else pd.DataFrame()
                            test_part = temp_part.iloc[len(temp_part)//2:]
                    else:
                        test_size = max(1, sample_count // 5)
                        val_size = max(1, sample_count // 10)
                        
                        shuffled_data = word_data.sample(frac=1, random_state=42).reset_index(drop=True)
                        
                        test_part = shuffled_data.iloc[:test_size]
                        val_part = shuffled_data.iloc[test_size:test_size + val_size]
                        train_part = shuffled_data.iloc[test_size + val_size:]
                    
                    train_data.append(train_part)
                    if not val_part.empty:
                        val_data.append(val_part)
                    test_data.append(test_part)
                    
                    print(f"{word}: {sample_count} samples → "
                          f"training ({len(train_part)}) + "
                          f"validation ({len(val_part) if not val_part.empty else 0}) + "
                          f"test ({len(test_part)})")
                    
                except Exception as e:
                    print(f"Warning: Failed to split {word}: {e} - adding to training")
                    train_data.append(word_data)
        
        final_splits = {
            'train': pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame(),
            'validation': pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame(),
            'test': pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        }
        
        print(f"\nFinal data split:")
        for split_name, split_df in final_splits.items():
            if not split_df.empty:
                unique_words = len(split_df['word'].unique())
                unique_speakers = len(split_df['speaker'].unique())
                print(f"  {split_name}: {len(split_df)} samples, "
                      f"{unique_words} words, {unique_speakers} speakers")
            else:
                print(f"  {split_name}: empty")
        
        return final_splits