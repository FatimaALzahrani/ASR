import pandas as pd


class FairSplitter:
    def __init__(self, test_size=0.2, val_size=0.1):
        self.test_size = test_size
        self.val_size = val_size
    
    def create_fair_splits(self, all_data):
        print(f"Creating fair data splits...")
        print("=" * 50)
        
        train_data = []
        val_data = []
        test_data = []
        
        print("Splitting by speaker:")
        
        for speaker in all_data['speaker'].unique():
            speaker_data = all_data[all_data['speaker'] == speaker].copy()
            n_samples = len(speaker_data)
            
            print(f"\n{speaker}: {n_samples} samples")
            
            if n_samples < 3:
                train_data.append(speaker_data)
                print(f"  -> Training: {n_samples}, Validation: 0, Test: 0")
                continue
            
            speaker_train = []
            speaker_val = []
            speaker_test = []
            
            for word in speaker_data['word'].unique():
                word_data = speaker_data[speaker_data['word'] == word]
                word_count = len(word_data)
                
                if word_count == 1:
                    speaker_train.append(word_data)
                elif word_count == 2:
                    speaker_train.append(word_data.iloc[:1])
                    speaker_test.append(word_data.iloc[1:])
                elif word_count == 3:
                    speaker_train.append(word_data.iloc[:2])
                    speaker_test.append(word_data.iloc[2:])
                else:
                    word_data_shuffled = word_data.sample(frac=1, random_state=42).reset_index(drop=True)
                    
                    test_count = max(1, int(word_count * self.test_size))
                    val_count = max(1, int(word_count * self.val_size)) if word_count > 5 else 0
                    train_count = word_count - test_count - val_count
                    
                    speaker_train.append(word_data_shuffled.iloc[:train_count])
                    if val_count > 0:
                        speaker_val.append(word_data_shuffled.iloc[train_count:train_count + val_count])
                    speaker_test.append(word_data_shuffled.iloc[train_count + val_count:])
            
            if speaker_train:
                speaker_train_df = pd.concat(speaker_train, ignore_index=True)
                train_data.append(speaker_train_df)
            
            if speaker_val:
                speaker_val_df = pd.concat(speaker_val, ignore_index=True)
                val_data.append(speaker_val_df)
            
            if speaker_test:
                speaker_test_df = pd.concat(speaker_test, ignore_index=True)
                test_data.append(speaker_test_df)
            
            train_count = len(speaker_train_df) if speaker_train else 0
            val_count = len(speaker_val_df) if speaker_val else 0
            test_count = len(speaker_test_df) if speaker_test else 0
            
            print(f"  -> Training: {train_count}, Validation: {val_count}, Test: {test_count}")
        
        final_train = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        final_val = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
        final_test = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        return final_train, final_val, final_test