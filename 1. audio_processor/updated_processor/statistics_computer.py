import pandas as pd
from typing import Dict


class StatisticsComputer:
    def calculate_statistics(self, splits: Dict[str, pd.DataFrame]) -> Dict:
        all_data = pd.concat(splits.values(), ignore_index=True)
        
        stats = {
            'total_samples': len(all_data),
            'total_duration': all_data['duration'].sum(),
            'avg_duration': all_data['duration'].mean(),
            'min_duration': all_data['duration'].min(),
            'max_duration': all_data['duration'].max(),
            'words': {
                'total': len(all_data['word'].unique()),
                'distribution': all_data['word'].value_counts().to_dict()
            },
            'speakers': {
                'total': len(all_data['speaker'].unique()),
                'distribution': all_data['speaker'].value_counts().to_dict()
            },
            'quality_levels': {
                'distribution': all_data['quality'].value_counts().to_dict(),
                'percentages': (all_data['quality'].value_counts() / len(all_data) * 100).to_dict()
            },
            'splits': {
                split_name: {
                    'samples': len(split_data),
                    'words': len(split_data['word'].unique()),
                    'speakers': len(split_data['speaker'].unique()),
                    'duration': split_data['duration'].sum()
                }
                for split_name, split_data in splits.items()
            }
        }
        
        return stats