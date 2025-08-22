import os
from pathlib import Path
from trainer import SpeechTrainer


class TrainingPipeline:
    
    def __init__(self):
        os.makedirs('models', exist_ok=True)
        self.results = {}
        
    def train_general_model(self, epochs=60):
        print("=" * 50)
        print("Training General Model")
        print("=" * 50)
        
        general_trainer = SpeechTrainer('general')
        general_model, train_losses, val_losses, general_report = \
            general_trainer.train_model(epochs=epochs)
        
        self.results['general'] = general_report
        print(f"General model training completed. Accuracy: {general_report['accuracy']:.3f}")
        
        return general_model, general_report
    
    def train_personalized_models(self, epochs=60):
        """Train personalized models for each speaker"""
        print("=" * 50)
        print("Training Personalized Models")
        print("=" * 50)
        
        speakers = ['أحمد', 'عاصم', 'هيفاء', 'أسيل', 'وسام']
        personalized_models = {}
        
        for speaker in speakers:
            speaker_path = Path(f'processed_data/speakers/{speaker}')
            
            if speaker_path.exists():
                print(f"Training model for speaker: {speaker}")
                trainer = SpeechTrainer('personalized')
                model, train_losses, val_losses, report = \
                    trainer.train_model(speaker, epochs=epochs)
                
                personalized_models[speaker] = model
                self.results[speaker] = report
                
                print(f"Model for {speaker} completed. Accuracy: {report['accuracy']:.3f}")
            else:
                print(f"No data found for speaker: {speaker}")
        
        return personalized_models
    
    def print_results_summary(self):
        """Print summary of all training results"""
        print("\n" + "=" * 50)
        print("Training Results Summary")
        print("=" * 50)
        
        if not self.results:
            print("No training results available.")
            return
        
        # Sort results by accuracy (descending)
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        for model_name, report in sorted_results:
            print(f"{model_name:15}: {report['accuracy']:.3f}")
        
        # Find best performing model
        best_model, best_report = sorted_results[0]
        print(f"\nBest performing model: {best_model} ({best_report['accuracy']:.3f})")
    
    def run_full_training(self, general_epochs=60, personalized_epochs=60):
        """Run complete training pipeline"""
        print("Starting Speech Recognition Training Pipeline")
        print("=" * 60)
        
        # Train general model
        general_model, general_report = self.train_general_model(general_epochs)
        
        # Train personalized models
        personalized_models = self.train_personalized_models(personalized_epochs)
        
        # Print summary
        self.print_results_summary()
        
        print("\nTraining pipeline completed successfully!")
        return self.results
    
    def get_results(self):
        return self.results