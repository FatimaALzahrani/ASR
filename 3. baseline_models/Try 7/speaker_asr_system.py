from config import *
from data_loader import DataLoader
from model_trainer import ModelTrainer
from evaluator import Evaluator

class SpeakerASRSystem:
    def __init__(self, sample_rate=Config.SAMPLE_RATE, duration=Config.DURATION, random_state=Config.RANDOM_STATE):
        self.sample_rate = sample_rate
        self.duration = duration
        self.random_state = random_state
        
        self.data_loader = DataLoader(random_state)
        self.model_trainer = ModelTrainer(random_state)
        self.evaluator = Evaluator()
        
        self.processed_data = None
        self.speaker_models = {}
        self.global_models = {}
        
        print("Speaker-Specific ASR System initialized")
    
    def run_complete_pipeline(self, data_path, output_dir='output'):
        print("Starting Speaker-Specific ASR Pipeline...")
        print("=" * 70)
        
        try:
            self.processed_data = self.data_loader.process_complete_dataset(data_path)
            
            self.speaker_models = self.model_trainer.train_speaker_specific_models(self.processed_data)
            
            self.global_models = self.model_trainer.train_global_models(self.processed_data)
            
            results = self.evaluator.evaluate_all_models(self.speaker_models, self.global_models)
            
            results_file, summary_file = self.evaluator.save_results(output_dir)
            
            print("=" * 70)
            print("SPEAKER-SPECIFIC ASR PIPELINE COMPLETED!")
            print("=" * 70)
            print(f"Results saved to: {output_dir}")
            print("=" * 70)
            
            return results
            
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            raise