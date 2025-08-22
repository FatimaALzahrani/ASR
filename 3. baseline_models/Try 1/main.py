# python main.py training_dataset.csv

import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from model_evaluator import ModelEvaluator
from audio_analyzer import AudioAnalyzer
from report_generator import ReportGenerator
from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleEvaluator:
    def __init__(self, data_path=None):
        if not data_path:
            if len(sys.argv) > 1:
                data_path = sys.argv[1]
            elif Config.DEFAULT_DATA_PATH:
                data_path = Config.DEFAULT_DATA_PATH
            else:
                raise ValueError(
                    "Data path is required. Please provide it in one of these ways:\n"
                    "1. Pass as argument: python main.py path/to/my/data.csv\n"
                    "2. Set DEFAULT_DATA_PATH in config.py\n"
                    "3. Pass to SimpleEvaluator(data_path='path/to/data.csv')"
                )
        
        self.data_path = data_path
        self.device = device
        
        print(f"Using data file: {self.data_path}")
        
        self.data_loader = DataLoader(self.data_path)
        self.feature_extractor = FeatureExtractor()
        self.model_evaluator = ModelEvaluator()
        self.audio_analyzer = AudioAnalyzer()
        self.report_generator = ReportGenerator()
        
        self.data_loader.load_and_analyze_data()
        
    def run_evaluation(self):
        try:
            print("Starting comprehensive evaluation...")
            print("=" * 60)
            
            df = self.data_loader.get_filtered_data()
            
            features_array = self.feature_extractor.extract_features_batch(df['file_path'])
            
            labels_list = df['word'].tolist()
            speakers_list = df['speaker'].tolist()
            
            baseline_results = self.model_evaluator.evaluate_baseline_models(features_array, labels_list)
            
            speaker_analysis = self.audio_analyzer.analyze_by_speaker(features_array, labels_list, speakers_list)
            
            word_analysis = self.audio_analyzer.analyze_by_word(labels_list)
            
            dataset_info = self.data_loader.get_data_info()
            
            report = self.report_generator.generate_comprehensive_report(
                dataset_info, baseline_results, speaker_analysis, word_analysis
            )
            
            self.report_generator.print_final_results(baseline_results, speaker_analysis)
            
            return report
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("Attempting to continue with available results...")
            import traceback
            traceback.print_exc()
            return None

def main():
    print("Simple Model Evaluator for Real Data")
    print("=" * 60)
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python main.py <path_to_csv_file>")
            print("  python main.py path/to/training_dataset.csv")
            print("\nExample:")
            print("  python main.py data/training_dataset.csv")
            return
        
        evaluator = SimpleEvaluator()
        
        report = evaluator.run_evaluation()
        
        if report:
            print("\nEvaluation completed successfully!")
        else:
            print("\nEvaluation completed with some issues. Check the output above for details.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease check if the file path is correct and the file exists.")
        
    except ValueError as e:
        print(f"Error: {e}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check your data file format and try again.")

if __name__ == "__main__":
    main()