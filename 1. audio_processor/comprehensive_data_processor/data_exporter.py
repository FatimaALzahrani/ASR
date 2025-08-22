import json


class DataExporter:
    def __init__(self, dataframe, speaker_manager, statistics):
        self.df = dataframe
        self.speaker_manager = speaker_manager
        self.statistics = statistics
    
    def export_training_data(self):
        print("Exporting training data...")
        
        training_data = self.df[['file_path', 'word', 'speaker', 'duration', 'quality_score']].copy()
        training_data.to_csv('training_dataset.csv', index=False, encoding='utf-8')
        
        metadata = {
            'speaker_mapping': self.speaker_manager.speaker_mapping,
            'speaker_info': self.speaker_manager.speaker_info,
            'statistics': self.statistics
        }
        
        with open('training_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print("Training data exported successfully")