import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from data_scanner import DataScanner
from data_balancer import DataBalancer
from data_splitter import DataSplitter
from visualization import DataVisualizer
from data_validator import DataValidator
from utils import safe_json_save, convert_to_json_serializable, print_processing_summary
import config


class DownSyndromeDataProcessor:
    
    def __init__(self, data_root: str = config.DATA_ROOT, skip_validation: bool = False):
        self.data_root = Path(data_root)
        self.processed_dir = self.data_root / config.PROCESSED_DIR
        self.reports_dir = self.data_root / config.REPORTS_DIR
        self.skip_validation = skip_validation
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.scanner = DataScanner(data_root)
        self.balancer = DataBalancer()
        self.splitter = DataSplitter()
        self.visualizer = DataVisualizer(self.reports_dir)
        self.validator = DataValidator()
        
        # Results storage
        self.df = None
        self.balance_info = None
        self.splits = None
        self.failed_files = None
    
    def run_complete_analysis(self) -> bool:
        print("Starting comprehensive Down Syndrome speech data analysis...")
        print("=" * 60)
        
        try:
            # 1. Scan audio files
            success = self._scan_files()
            if not success:
                return False
            
            # 2. Validate dataset
            self._validate_dataset()
            
            # 3. Analyze data balance
            self._analyze_balance()
            
            # 4. Create data splits
            self._create_splits()
            
            # 5. Generate visualizations
            try:
                self._create_visualizations()
            except Exception as e:
                print(f"Warning: Visualization generation failed: {e}")
                print("Continuing without visualizations...")
            
            # 6. Save processed data
            save_success = self._save_results()
            
            # 7. Print summary
            self._print_final_summary()
            
            # Return success if main data was saved
            return save_success
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            
            # Try to save whatever we have
            if hasattr(self, 'splits') and self.splits:
                print("Attempting to save partial results...")
                try:
                    self._save_data_splits()
                    print("Partial data splits saved successfully")
                    return True
                except:
                    print("Failed to save even partial results")
            
            return False
    
    def _scan_files(self) -> bool:
        self.df, self.failed_files = self.scanner.scan_audio_files()
        
        if self.df.empty:
            print("No valid data found for processing")
            return False
        
        print(f"Successfully loaded {len(self.df)} audio files")
        
        # Save failed files report
        if self.failed_files:
            failed_df = pd.DataFrame(self.failed_files)
            failed_df.to_csv(self.reports_dir / 'failed_files.csv', 
                           index=False, encoding='utf-8')
            print(f"Failed files report saved ({len(self.failed_files)} files)")
        
        return True
    
    def _validate_dataset(self):
        try:
            validation_report = self.validator.validate_dataset(self.df)
            self.validator.print_report(validation_report)
            
            # Save validation report
            validation_file = self.reports_dir / 'validation_report.json'
            safe_json_save(validation_report, validation_file)
            
            # Get splitting advice
            splitting_advice = self.validator.get_splitting_strategy_advice(self.df)
            if splitting_advice:
                print("\nSplitting Strategy Advice:")
                for advice in splitting_advice['advice']:
                    print(f"  • {advice}")
                
                # Save splitting advice
                advice_file = self.reports_dir / 'splitting_advice.json'
                safe_json_save(splitting_advice, advice_file)
            
            # Critical issues check
            if validation_report['issues'] and not self.skip_validation:
                print("\n⚠️  Critical issues detected. Review before proceeding.")
                response = input("Continue anyway? (y/N): ").lower().strip()
                if response != 'y':
                    raise ValueError("Processing aborted due to critical issues")
            elif validation_report['issues'] and self.skip_validation:
                print("\n⚠️  Critical issues detected but continuing due to --skip-validation flag")
                
        except Exception as e:
            print(f"Warning: Validation failed: {e}")
            print("Continuing with processing...")
    
    def _analyze_balance(self):
        self.balance_info = self.balancer.analyze_data_balance(self.df)
    
    def _create_splits(self):
        # Filter rare words before splitting
        filtered_df = self.balancer.filter_rare_words(self.df, min_samples=1)  # Keep all words
        
        # Create general splits
        self.splits = self.splitter.create_balanced_splits(filtered_df)
        
        # Create speaker-specific splits with error handling
        try:
            speaker_splits = self.splitter.create_speaker_specific_splits(filtered_df)
            if speaker_splits:
                self._save_speaker_splits(speaker_splits)
            else:
                print("Warning: No speaker-specific splits created")
        except Exception as e:
            print(f"Warning: Failed to create speaker-specific splits: {e}")
            print("Continuing with general splits only")
    
    def _create_visualizations(self):
        self.visualizer.create_analysis_plots(self.df, self.balance_info)
        self.visualizer.create_balance_plots(self.balance_info)
        self.visualizer.create_quality_analysis_plots(self.df)
    
    def _save_results(self):
        print("Saving processed data...")
        
        success_count = 0
        total_operations = 3  # splits, mappings, reports
        
        # Save data splits
        try:
            self._save_data_splits()
            success_count += 1
            print("✅ Data splits saved successfully")
        except Exception as e:
            print(f"❌ Error saving data splits: {e}")
        
        # Save mappings
        try:
            self._save_mappings()
            success_count += 1
            print("✅ Mappings saved successfully")
        except Exception as e:
            print(f"❌ Error saving mappings: {e}")
        
        # Save reports
        try:
            self._save_reports()
            success_count += 1
            print("✅ Reports saved successfully")
        except Exception as e:
            print(f"❌ Error saving reports: {e}")
        
        # Summary
        if success_count == total_operations:
            print(f"🎉 All results saved successfully to: {self.processed_dir}")
        elif success_count > 0:
            print(f"⚠️ Partial success: {success_count}/{total_operations} operations completed")
            print(f"Main data files should be available in: {self.processed_dir}")
        else:
            print(f"❌ Failed to save results. Check permissions and disk space.")
            print(f"Target directory: {self.processed_dir}")
        
        return success_count > 0
    
    def _save_data_splits(self):
        if not self.splits:
            print("Warning: No splits to save")
            return
            
        saved_splits = []
        for split_name, split_df in self.splits.items():
            if not split_df.empty:
                output_path = self.processed_dir / f"{split_name}.csv"
                split_df.to_csv(output_path, index=False, encoding='utf-8')
                saved_splits.append(split_name)
                print(f"  {split_name}: {len(split_df)} samples -> {output_path.name}")
            else:
                print(f"  {split_name}: empty - skipped")
        
        if not saved_splits:
            print("Warning: No split files were saved")
        elif 'train' not in saved_splits:
            print("WARNING: No training data was saved - this will prevent model training")
    
    def _save_speaker_splits(self, speaker_splits: Dict):
        if not speaker_splits:
            print("No speaker-specific splits to save")
            return
            
        speakers_dir = self.processed_dir / "speakers"
        speakers_dir.mkdir(exist_ok=True)
        
        saved_count = 0
        for speaker, splits in speaker_splits.items():
            try:
                speaker_dir = speakers_dir / speaker
                speaker_dir.mkdir(exist_ok=True)
                
                for split_name, split_df in splits.items():
                    if not split_df.empty:
                        output_path = speaker_dir / f"{split_name}.csv"
                        split_df.to_csv(output_path, index=False, encoding='utf-8')
                    else:
                        print(f"Warning: Empty {split_name} split for speaker {speaker}")
                
                saved_count += 1
            except Exception as e:
                print(f"Error saving splits for speaker {speaker}: {e}")
        
        if saved_count > 0:
            print(f"Speaker-specific splits saved for {saved_count} speakers in: {speakers_dir}")
        else:
            print("Warning: No speaker-specific splits were saved")
    
    def _save_mappings(self):
        all_data = pd.concat([df for df in self.splits.values() if not df.empty], 
                           ignore_index=True)
        
        word_to_id = {word: idx for idx, word in enumerate(sorted(all_data['word'].unique()))}
        speaker_to_id = {speaker: idx for idx, speaker in enumerate(sorted(all_data['speaker'].unique()))}
        
        mappings = {
            'word_to_id': word_to_id,
            'id_to_word': {str(v): k for k, v in word_to_id.items()},  # Convert keys to strings
            'speaker_to_id': speaker_to_id,
            'id_to_speaker': {str(v): k for k, v in speaker_to_id.items()},  # Convert keys to strings
            'num_words': int(len(word_to_id)),
            'num_speakers': int(len(speaker_to_id))
        }
        
        with open(self.processed_dir / 'mappings.json', 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        return mappings
    
    def _save_reports(self):
        balance_info_json = self._convert_to_json_serializable(self.balance_info)
        
        with open(self.reports_dir / 'balance_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(balance_info_json, f, ensure_ascii=False, indent=2)
        
        # Processing statistics report
        processing_stats = self.scanner.get_processing_stats()
        
        # Calculate stats with proper type conversion
        data_stats = {
            'total_samples': int(len(self.df)),
            'total_words': int(len(self.df['word'].unique())),
            'total_speakers': int(len(self.df['speaker'].unique())),
            'avg_duration': float(self.df['duration'].mean()),
            'avg_quality_score': float(self.df['quality_score'].mean())
        }
        
        processing_report = {
            'processing_stats': processing_stats,
            'data_stats': data_stats,
            'speaker_info': config.SPEAKER_INFO,
            'excluded_words': config.EXCLUDED_WORDS,
            'config': {
                'target_sr': config.TARGET_SAMPLE_RATE,
                'target_duration': config.TARGET_DURATION,
                'min_duration': config.MIN_DURATION,
                'max_duration': config.MAX_DURATION
            }
        }
        
        with open(self.reports_dir / 'processing_report.json', 'w', encoding='utf-8') as f:
            json.dump(processing_report, f, ensure_ascii=False, indent=2)
    
    def _convert_to_json_serializable(self, obj):
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    def _print_final_summary(self):
        if self.df is not None and self.balance_info is not None:
            summary_stats = {
                'total_samples': len(self.df),
                'total_words': self.balance_info['total_words'],
                'total_speakers': len(self.df['speaker'].unique()),
                'avg_quality_score': float(self.df['quality_score'].mean()),
                'avg_duration': float(self.df['duration'].mean()),
                'processing_stats': self.scanner.get_processing_stats()
            }
            
            print_processing_summary(summary_stats, "Analysis Completed Successfully!")
        else:
            print("\n" + "=" * 60)
            print("Analysis completed with limited results")
            print("=" * 60)
        
        print(f"Output directories:")
        print(f"  Processed data: {self.processed_dir}")
        print(f"  Reports and plots: {self.reports_dir}")
        
        print("\nReady for next step: Audio enhancement and model training!")
    
    def get_summary_stats(self) -> Optional[Dict]:
        if self.df is None or self.balance_info is None:
            return None
        
        return {
            'total_samples': len(self.df),
            'total_words': self.balance_info['total_words'],
            'total_speakers': len(self.df['speaker'].unique()),
            'avg_quality_score': float(self.df['quality_score'].mean()),
            'avg_duration': float(self.df['duration'].mean()),
            'processing_stats': self.scanner.get_processing_stats()
        }