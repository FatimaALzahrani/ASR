import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


class ProcessedDataSaver:
    def __init__(self, processed_dir: Path, reports_dir: Path, 
                 excluded_words: List[str], numpy_converter):
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.excluded_words = excluded_words
        self.numpy_converter = numpy_converter
    
    def save_processed_data(self, splits: Dict[str, pd.DataFrame], 
                          balance_info: Dict, speaker_info: Dict) -> bool:
        print("\nSaving processed data...")
        
        try:
            for split_name, split_df in splits.items():
                if not split_df.empty:
                    output_path = self.processed_dir / f"{split_name}.csv"
                    split_df.to_csv(output_path, index=False, encoding='utf-8')
                    print(f"{split_name}: {len(split_df)} samples → {output_path}")
            
            all_data = pd.concat([df for df in splits.values() if not df.empty], ignore_index=True)
            
            if not all_data.empty:
                word_to_id = {word: idx for idx, word in enumerate(sorted(all_data['word'].unique()))}
                speaker_to_id = {speaker: idx for idx, speaker in enumerate(sorted(all_data['speaker'].unique()))}
                
                mappings = {
                    'word_to_id': word_to_id,
                    'id_to_word': {v: k for k, v in word_to_id.items()},
                    'speaker_to_id': speaker_to_id,
                    'id_to_speaker': {v: k for k, v in speaker_to_id.items()},
                    'num_words': len(word_to_id),
                    'num_speakers': len(speaker_to_id)
                }
                
                mappings_path = self.processed_dir / 'mappings.json'
                with open(mappings_path, 'w', encoding='utf-8') as f:
                    json.dump(mappings, f, ensure_ascii=False, indent=2)
                print(f"Data mappings → {mappings_path}")
                
                balance_path = self.reports_dir / 'balance_analysis.json'
                with open(balance_path, 'w', encoding='utf-8') as f:
                    json.dump(balance_info, f, ensure_ascii=False, indent=2)
                print(f"Balance analysis → {balance_path}")
                
                processing_report = {
                    'data_summary': {
                        'total_samples': int(len(all_data)),
                        'total_words': len(word_to_id),
                        'total_speakers': len(speaker_to_id),
                        'avg_duration': float(all_data['duration'].mean()),
                        'avg_quality_score': float(all_data['quality_score'].mean())
                    },
                    'speaker_info': self.numpy_converter.convert_numpy_types(speaker_info),
                    'excluded_words': self.excluded_words,
                    'split_statistics': {
                        split_name: {
                            'samples': int(len(split_df)),
                            'words': int(len(split_df['word'].unique())),
                            'speakers': int(len(split_df['speaker'].unique()))
                        }
                        for split_name, split_df in splits.items() if not split_df.empty
                    }
                }
                
                report_path = self.reports_dir / 'processing_report.json'
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(processing_report, f, ensure_ascii=False, indent=2)
                print(f"Comprehensive report → {report_path}")
                
                return True
            else:
                print("Error: No data to save")
                return False
                
        except Exception as e:
            print(f"Error saving data: {e}")
            return False