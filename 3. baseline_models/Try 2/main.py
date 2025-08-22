import os
import torch
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from whisper_evaluator import WhisperEvaluator
from result_analyzer import ResultAnalyzer
from results_saver import ResultsSaver
from config import Config

class SimpleTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.data_loader = DataLoader()
        self.evaluator = WhisperEvaluator(self.device)
        self.analyzer = ResultAnalyzer(self.evaluator)
        self.results_saver = ResultsSaver()
        
        self.data_loader.load_data()
    
    def run_comprehensive_evaluation(self) -> dict:
        print("Starting comprehensive evaluation...")
        
        results = {
            "model_info": {
                "model_name": Config.WHISPER_MODEL_NAME,
                "device": str(self.device),
                "language": Config.LANGUAGE
            },
            "data_info": self.data_loader.get_data_info()
        }
        
        print("\n" + "="*50)
        sample_results = self.evaluator.evaluate_sample(
            self.data_loader.get_sample_data(), 
            "Quick sample"
        )
        results["sample_evaluation"] = sample_results
        
        print("\n" + "="*50)
        test_results = self.evaluator.evaluate_sample(
            self.data_loader.get_test_data(), 
            "Test data"
        )
        results["test_evaluation"] = test_results
        
        print("\n" + "="*50)
        speaker_results = self.analyzer.analyze_by_speaker(
            self.data_loader.get_sample_data()
        )
        results["speaker_analysis"] = speaker_results
        
        print("\n" + "="*50)
        word_results = self.analyzer.analyze_by_word(
            self.data_loader.get_sample_data()
        )
        results["word_analysis"] = word_results
        
        return results

def main():
    try:
        trainer = SimpleTrainer()
        
        results = trainer.run_comprehensive_evaluation()
        
        trainer.results_saver.save_results(results)
        
        print("\nEvaluation completed successfully!")
        print("Check the results/ folder for detailed results")
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Please check your data directory path in config.py")
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all requirements: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        print("Run 'python debug_utils.py' to validate your setup")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()