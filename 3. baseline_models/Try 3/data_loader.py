import pandas as pd


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.train_speakers = None
        self.test_speakers = None
        
    def load_data(self):
        print("Loading corrected data...")
        try:
            self.data = pd.read_csv(self.file_path)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def split_by_speakers(self):
        speaker_counts = self.data['speaker'].value_counts()
        speakers = speaker_counts.index.tolist()
        
        self.test_speakers = ['أحمد']
        self.train_speakers = [s for s in speakers if s not in self.test_speakers]
        
        train_mask = self.data['speaker'].isin(self.train_speakers)
        test_mask = self.data['speaker'].isin(self.test_speakers)
        
        train_data = self.data[train_mask]
        test_data = self.data[test_mask]
        
        return train_data, test_data