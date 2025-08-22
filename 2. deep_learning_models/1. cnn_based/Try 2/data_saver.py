import pandas as pd
import json
from pathlib import Path


class DataSaverFinal:
    def __init__(self, processed_dir):
        self.processed_dir = Path(processed_dir)
    
    def save_data(self, general_splits, speaker_splits):
        """Save both general and speaker-specific data"""
        # Save general data
        for split, data in general_splits.items():
            if not data.empty:
                data.to_csv(self.processed_dir / f"{split}.csv", index=False, encoding='utf-8')
        
        # Save speaker-specific data
        speaker_dir = self.processed_dir / "speakers"
        speaker_dir.mkdir(exist_ok=True)
        
        for speaker, splits in speaker_splits.items():
            speaker_path = speaker_dir / speaker
            speaker_path.mkdir(exist_ok=True)
            for split, data in splits.items():
                if not data.empty:
                    data.to_csv(speaker_path / f"{split}.csv", index=False, encoding='utf-8')
        
        # Create data mappings
        all_data = pd.concat([d for d in general_splits.values() if not d.empty])
        mappings = {
            'word_to_id': {w: i for i, w in enumerate(sorted(all_data['word'].unique()))},
            'speaker_to_id': {s: i for i, s in enumerate(sorted(all_data['speaker'].unique()))},
            'num_words': len(all_data['word'].unique()),
            'num_speakers': len(all_data['speaker'].unique())
        }
        mappings['id_to_word'] = {v: k for k, v in mappings['word_to_id'].items()}
        mappings['id_to_speaker'] = {v: k for k, v in mappings['speaker_to_id'].items()}
        
        with open(self.processed_dir / 'mappings.json', 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        print(f"Data saved:")
        print(f"   General: {len(general_splits['train'])} train, {len(general_splits.get('validation', []))} validation, {len(general_splits.get('test', []))} test")
        print(f"   Personalized: {len(speaker_splits)} speakers × train/test")