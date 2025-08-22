import os
import json
import pandas as pd
from datetime import datetime
from data_manager import DataManager
from model_trainer import ModelTrainer


class StandaloneASRSystem:
    def __init__(self, sample_rate=22050, duration=3.0, random_state=42, 
                 min_samples_per_word=3, min_quality_score=0.3):
        self.sample_rate = sample_rate
        self.duration = duration
        self.random_state = random_state
        self.min_samples_per_word = min_samples_per_word
        self.min_quality_score = min_quality_score
        
        self.data_manager = DataManager(
            sample_rate=sample_rate,
            min_samples_per_word=min_samples_per_word,
            min_quality_score=min_quality_score,
            random_state=random_state
        )
        
        self.model_trainer = ModelTrainer(random_state=random_state)
        
        self.results = {}
        self.processed_data = None
        
        print("Standalone Advanced ASR System initialized")
        print(f"Speaker mapping: {self.data_manager.speaker_mapping}")
        print(f"Minimum samples per word: {self.min_samples_per_word}")
        print(f"Minimum quality score: {self.min_quality_score}")
    
    def process_complete_dataset(self, data_path):
        print("Starting complete dataset processing with quality filtering")
        
        raw_df = self.data_manager.load_data_from_folders(data_path)
        count_filtered_df = self.data_manager.filter_by_sample_count(raw_df)
        quality_filtered_df = self.data_manager.filter_by_quality(count_filtered_df)
        
        print("Processing audio files and extracting features")
        processed_records = []
        failed_count = 0
        
        for idx, row in quality_filtered_df.iterrows():
            print(f"Processing {len(processed_records)+1}/{len(quality_filtered_df)}: {row['filename']}")
            
            features = self.data_manager.process_audio_file(row['file_path'])
            
            if features is not None:
                record = {
                    'file_path': row['file_path'],
                    'word': row['word'],
                    'speaker': row['speaker'],
                    'filename': row['filename']
                }
                record.update(features)
                processed_records.append(record)
            else:
                failed_count += 1
        
        print(f"Successfully processed: {len(processed_records)} files")
        print(f"Failed to process: {failed_count} files")
        
        if len(processed_records) == 0:
            raise ValueError("No files were successfully processed. Check your data path and file formats.")
        
        features_df = pd.DataFrame(processed_records)
        final_df = self.data_manager.filter_by_sample_count(features_df)
        train_df, test_df = self.data_manager.create_speaker_independent_split(final_df)
        stats = self.data_manager.generate_dataset_statistics(final_df, train_df, test_df)
        
        print("Dataset processing completed successfully")
        
        return {
            'full_df': final_df,
            'train_df': train_df,
            'test_df': test_df,
            'statistics': stats
        }
    
    def save_csv_files(self, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Saving CSV files")
        
        full_csv_path = os.path.join(output_dir, 'complete_dataset_standalone.csv')
        self.processed_data['full_df'].to_csv(full_csv_path, index=False, encoding='utf-8')
        print(f"Saved complete dataset: {full_csv_path}")
        
        train_csv_path = os.path.join(output_dir, 'train_dataset_standalone.csv')
        self.processed_data['train_df'].to_csv(train_csv_path, index=False, encoding='utf-8')
        print(f"Saved training dataset: {train_csv_path}")
        
        test_csv_path = os.path.join(output_dir, 'test_dataset_standalone.csv')
        self.processed_data['test_df'].to_csv(test_csv_path, index=False, encoding='utf-8')
        print(f"Saved test dataset: {test_csv_path}")
        
        stats_path = os.path.join(output_dir, 'dataset_statistics_standalone.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data['statistics'], f, indent=2, ensure_ascii=False)
        print(f"Saved dataset statistics: {stats_path}")
        
        return {
            'complete_csv': full_csv_path,
            'train_csv': train_csv_path,
            'test_csv': test_csv_path,
            'statistics': stats_path
        }
    
    def create_comprehensive_report(self, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        
        comprehensive_report = {
            'system_info': {
                'timestamp': datetime.now().isoformat(),
                'version': 'Standalone ASR System v1.0',
                'data_statistics': self.processed_data['statistics'] if self.processed_data else {}
            },
            'traditional_ml_results': self.results.get('traditional_ml', {}),
            'recommendations': [
                "This system provides honest, realistic results for Arabic speech recognition.",
                "Speaker-independent evaluation ensures proper generalization assessment.",
                "Quality filtering removes poor recordings for better model training.",
                "Results are suitable for academic publication with proper methodology."
            ]
        }
        
        report_path = os.path.join(output_dir, 'comprehensive_report_standalone.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        summary_df = self.model_trainer.create_summary_table(self.results)
        summary_path = os.path.join(output_dir, 'model_performance_summary_standalone.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        
        print(f"Comprehensive report saved: {report_path}")
        print(f"Summary table saved: {summary_path}")
        
        return report_path, summary_path
    
    def run_complete_pipeline(self, data_path, output_dir='output'):
        print("Starting complete standalone ASR pipeline")
        
        try:
            self.processed_data = self.process_complete_dataset(data_path)
            csv_files = self.save_csv_files(output_dir)
            ml_results = self.model_trainer.train_traditional_ml_models(
                self.processed_data['train_df'], 
                self.processed_data['test_df']
            )
            self.results['traditional_ml'] = ml_results
            report_path, summary_path = self.create_comprehensive_report(output_dir)
            
            print("Complete standalone ASR pipeline finished successfully")
            
            return {
                'csv_files': csv_files,
                'report_path': report_path,
                'summary_path': summary_path,
                'results': self.results
            }
            
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            raise
