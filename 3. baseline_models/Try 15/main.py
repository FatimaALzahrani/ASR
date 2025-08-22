import os
from pathlib import Path

try:
    import librosa
    from scipy import signal
    import soundfile as sf
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from sklearn.model_selection import train_test_split
    import numpy as np
    print("All required libraries loaded successfully")
except ImportError as e:
    print(f"Installing required libraries: {e}")
    os.system("pip install librosa scipy soundfile transformers scikit-learn torch torchaudio numpy")

from trainer import UltimateTrainer


def main():
    print("Memory-Optimized Speech Recognition Trainer")
    print("=" * 60)
    
    data_path = input("Enter audio data folder path (or press Enter for auto-detect): ").strip()
    
    if not data_path:
        possible_paths = [
            "C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Codes/Real Codes/01_data_processing2/data/clean",
            "/home/ubuntu/clean", 
            "./upload",
            "./clean",
            "./data",
            "."
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                data_path = path
                print(f"Auto-detected path: {data_path}")
                break
    
    if not Path(data_path).exists():
        print(f"Path not found: {data_path}")
        return
    
    trainer = UltimateTrainer(data_path)
    
    results = trainer.run_complete_training()
    
    if results:
        print("\nFinal Results Summary:")
        print(f"Overall accuracy: {results['overall_accuracy']:.1%}")
        print(f"Total samples: {results.get('total_samples', 0)}")
        
        if results['speaker_analysis']:
            print("\nSpeaker Performance:")
            for speaker, data in results['speaker_analysis'].items():
                print(f"  {speaker}: {data['accuracy']:.1%} ({data['samples']} samples)")


if __name__ == "__main__":
    main()