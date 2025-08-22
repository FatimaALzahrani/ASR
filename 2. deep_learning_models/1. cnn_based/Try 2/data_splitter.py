import pandas as pd


class DataSplitterFinal:
    def create_splits(self, df):
        """Create both general and speaker-specific splits"""
        general_splits = self.split_general(df)
        speaker_splits = self.split_by_speaker(df)
        return general_splits, speaker_splits

    def split_general(self, df):
        """Create general splits for multi-speaker model"""
        train_data, val_data, test_data = [], [], []
        
        for word in df['word'].unique():
            word_data = df[df['word'] == word].sample(frac=1, random_state=42)
            n = len(word_data)
            
            if n >= 10:
                train_end = int(0.7 * n)
                val_end = int(0.85 * n)
            elif n >= 5:
                train_end = n - 2
                val_end = n - 1
            else:
                train_end = n - 1
                val_end = n
            
            train_data.append(word_data.iloc[:train_end])
            if val_end > train_end:
                val_data.append(word_data.iloc[train_end:val_end])
            if n > val_end:
                test_data.append(word_data.iloc[val_end:])
        
        return {
            'train': pd.concat(train_data, ignore_index=True),
            'validation': pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame(),
            'test': pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        }

    def split_by_speaker(self, df):
        """Create speaker-specific splits for personalized models"""
        speaker_data = {}
        
        for speaker in df['speaker'].unique():
            speaker_df = df[df['speaker'] == speaker]
            
            train_speaker = []
            test_speaker = []
            
            for word in speaker_df['word'].unique():
                word_data = speaker_df[speaker_df['word'] == word]
                n = len(word_data)
                
                if n == 1:
                    train_speaker.append(word_data)
                else:
                    test_size = max(1, n // 5)
                    shuffled = word_data.sample(frac=1, random_state=42)
                    test_speaker.append(shuffled.iloc[:test_size])
                    train_speaker.append(shuffled.iloc[test_size:])
            
            speaker_data[speaker] = {
                'train': pd.concat(train_speaker, ignore_index=True) if train_speaker else pd.DataFrame(),
                'test': pd.concat(test_speaker, ignore_index=True) if test_speaker else pd.DataFrame()
            }
        
        return speaker_data