import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path
from numpy_type_converter import NumpyTypeConverter
from speaker_identifier import SpeakerIdentifierFixed
from audio_quality_analyzer import AudioQualityAnalyzerFixed
from audio_file_scanner import AudioFileScannerFixed
from data_balance_analyzer import DataBalanceAnalyzer
from improved_data_splitter import ImprovedDataSplitter
from data_visualizer import DataVisualizerFixed
from processed_data_saver import ProcessedDataSaver


class DownSyndromeProcessor:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.clean_audio_dir = self.data_root / "C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean"
        self.processed_dir = self.data_root / "processed"
        self.reports_dir = self.data_root / "reports"
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_sr = 16000
        self.target_duration = 3.0
        self.min_duration = 0.3
        self.max_duration = 15.0
        
        self.excluded_words = ['sleep']
        self.min_samples_per_word = 1
        
        self.numpy_converter = NumpyTypeConverter()
        self.speaker_identifier = SpeakerIdentifierFixed()
        self.quality_analyzer = AudioQualityAnalyzerFixed()
        self.file_scanner = AudioFileScannerFixed(
            self.clean_audio_dir, self.reports_dir, 
            self.excluded_words, self.min_duration, self.max_duration
        )
        self.balance_analyzer = DataBalanceAnalyzer(self.numpy_converter)
        self.data_splitter = ImprovedDataSplitter()
        self.visualizer = DataVisualizerFixed(self.reports_dir)
        self.data_saver = ProcessedDataSaver(
            self.processed_dir, self.reports_dir, 
            self.excluded_words, self.numpy_converter
        )
    
    def run_complete_analysis(self) -> bool:
        print("Starting comprehensive fixed analysis for Down syndrome data...")
        print("=" * 70)
        
        try:
            # 1. Scan files
            df = self.file_scanner.scan_audio_files(
                self.speaker_identifier, self.quality_analyzer, self.numpy_converter
            )
            if df.empty:
                print("Error: No valid data for processing")
                return False
            
            # 2. Analyze balance
            balance_info = self.balance_analyzer.analyze_data_balance(df)
            
            # 3. Create improved splits
            splits = self.data_splitter.create_improved_splits(df)
            
            # 4. Create visualizations
            self.visualizer.create_visualizations(df, balance_info)
            
            # 5. Save data
            success = self.data_saver.save_processed_data(
                splits, balance_info, self.speaker_identifier.speaker_info
            )
            
            if success:
                print("\n" + "=" * 70)
                print("Fixed analysis completed successfully!")
                
                all_data = pd.concat([df for df in splits.values() if not df.empty], ignore_index=True)
                print(f"Final statistics:")
                print(f"   • Total samples: {len(all_data)}")
                print(f"   • Number of words: {len(all_data['word'].unique())}")
                print(f"   • Number of speakers: {len(all_data['speaker'].unique())}")
                print(f"   • Average quality: {all_data['quality_score'].mean():.2f}")
                print(f"   • Average duration: {all_data['duration'].mean():.1f} seconds")
                
                print(f"\nGenerated files:")
                print(f"   • data/processed/ - Split data")
                print(f"   • data/reports/ - Reports and charts")
                
                return True
            else:
                return False
                
        except Exception as e:
            print(f"General analysis error: {e}")
            import traceback
            traceback.print_exc()
            return False