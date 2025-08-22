import json
import pandas as pd
from pathlib import Path
from typing import Dict


class DataSaver:
    def __init__(self, processed_dir: Path, data_dir: Path):
        self.processed_dir = processed_dir
        self.data_dir = data_dir
    
    def save_processed_data(self, splits: Dict[str, pd.DataFrame], 
                          mappings: Dict, statistics: Dict) -> None:
        for split_name, split_data in splits.items():
            output_path = self.processed_dir / f"{split_name}.csv"
            split_data.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Saved {split_name}: {output_path}")
        
        mappings_path = self.processed_dir / "mappings.json"
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        print(f"Saved mappings: {mappings_path}")
        
        stats_path = self.processed_dir / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        print(f"Saved statistics: {stats_path}")
        
        self._create_transcript_files(splits, mappings)
    
    def _create_transcript_files(self, splits: Dict[str, pd.DataFrame], mappings: Dict):
        all_data = pd.concat(splits.values(), ignore_index=True)
        transcripts_dir = self.data_dir / "C:/Users/فاطمة الزهراني/Desktop/ابحاث\الداون/Codes/Clean/1. audio_processor/updated_data_processor/output_files/transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        
        transcripts_data = all_data[['filename', 'speaker', 'text', 'quality', 'word']].copy()
        transcripts_data['word_id'] = transcripts_data['word'].map(mappings['word_to_id'])
        transcripts_path = transcripts_dir / "transcripts.csv"
        transcripts_data.to_csv(transcripts_path, index=False, encoding='utf-8')
        print(f"Saved transcripts: {transcripts_path}")
        
        speaker_info_data = all_data[['speaker', 'age', 'gender', 'iq_level', 'quality']].drop_duplicates()
        speaker_info_data.columns = ['speaker', 'age', 'gender', 'iq_level', 'speech_quality']
        speaker_info_path = transcripts_dir / "speaker_info.csv"
        speaker_info_data.to_csv(speaker_info_path, index=False, encoding='utf-8')
        print(f"Saved speaker info: {speaker_info_path}")