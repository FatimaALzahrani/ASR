import os
from pathlib import Path
from speech_trainer import SpeechTrainer


def train_all_models():
    os.makedirs('models', exist_ok=True)
    
    print("=" * 50)
    print("Training General Model")
    print("=" * 50)
    general_trainer = SpeechTrainer('general')
    general_model, _, _ = general_trainer.train_model()
    
    print("=" * 50)
    print("Training Personalized Models")
    print("=" * 50)
    speakers = ['Ahmed', 'Asem', 'Haifa', 'Aseel', 'Wessam']
    speaker_results = {}
    
    for speaker in speakers:
        speaker_path = Path(f'data/processed/speakers/{speaker}')
        if speaker_path.exists():
            trainer = SpeechTrainer('personalized')
            model, _, _ = trainer.train_model(speaker)
            speaker_results[speaker] = model
        else:
            print(f"Warning: No data found for speaker {speaker}")
    
    print("Training completed for all models!")
    return general_model, speaker_results


def main():
    print("Speech Recognition Training Pipeline")
    print("=" * 60)
    
    if not Path('data/processed/mappings.json').exists():
        print("Error: Processed data not found. Run data processing first.")
        return
    
    general_model, speaker_results = train_all_models()
    
    print(f"\nTraining Summary:")
    print(f"- General model: Trained")
    print(f"- Personalized models: {len(speaker_results)} speakers")
    
    print("\nModels saved in 'models/' directory")
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()