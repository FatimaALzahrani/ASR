import os
import sys
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from asr_system import ASRSystem


def main():
    print("Advanced Arabic Speech Recognition System")
    print("For Children with Down Syndrome Research")
    print("="*60)
    
    parser = argparse.ArgumentParser(description='Advanced ASR System with Deep Learning')
    parser.add_argument('--data_path', type=str, 
                       default="C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean",
                       help='Path to audio data directory')
    parser.add_argument('--output_path', type=str, default="output_files",
                       help='Path to save results')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate'], 
                       default='train', help='Operation mode')
    parser.add_argument('--audio_file', type=str, help='Audio file for prediction')
    parser.add_argument('--model_name', type=str, help='Model name for prediction')
    
    args = parser.parse_args()
    
    try:
        asr_system = ASRSystem(data_path=args.data_path, output_path=args.output_path)
        print(f"ASR system initialized successfully")
        print(f"Data path: {args.data_path}")
        print(f"Output path: {args.output_path}")
    except Exception as e:
        print(f"System initialization error: {e}")
        return
    
    if args.mode == 'train':
        print(f"\nStarting training mode...")
        train_system(asr_system)
        
    elif args.mode == 'predict':
        print(f"\nStarting prediction mode...")
        if not args.audio_file:
            print("Audio file required for prediction using --audio_file")
            return
        predict_audio(asr_system, args.audio_file, args.model_name)
        
    elif args.mode == 'evaluate':
        print(f"\nStarting evaluation mode...")
        evaluate_system(asr_system)


def train_system(asr_system):
    print("Starting deep learning model training...")
    
    try:
        start_time = time.time()
        results = asr_system.run_complete_evaluation()
        end_time = time.time()
        
        if results:
            best_model = results['absolute_best']
            print(f"\nTraining completed successfully!")
            print(f"Best model: {best_model['model']}")
            print(f"Best accuracy: {best_model['accuracy']*100:.2f}%")
            print(f"Results saved in: {asr_system.output_path}")
            print(f"Training time: {(end_time - start_time)/60:.2f} minutes")
            
            total_models = len(results['model_results'])
            successful_models = len([r for r in results['model_results'].values() if r > 0])
            
            print(f"\nTraining Statistics:")
            print(f"   Models trained: {total_models}")
            print(f"   Successful models: {successful_models}")
            print(f"   Success rate: {successful_models/total_models*100:.1f}%")
            
        else:
            print("Training failed!")
            return
            
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return


def predict_audio(asr_system, audio_file, model_name=None):
    print(f"Predicting audio file: {audio_file}")
    
    if not Path(audio_file).exists():
        print(f"File not found: {audio_file}")
        return
    
    if not asr_system.load_trained_models():
        print("Failed to load trained models")
        return
    
    try:
        result = asr_system.predict_word(audio_file, model_name)
        
        if result:
            print(f"\nPrediction Results:")
            print(f"   Predicted word: {result['predicted_word']}")
            print(f"   Confidence: {result['confidence']*100:.2f}%")
            print(f"   Model used: {result['model_used']}")
            
            probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
            print(f"\nTop 3 probabilities:")
            for i, (word, prob) in enumerate(probs[:3], 1):
                print(f"   {i}. {word}: {prob*100:.2f}%")
        else:
            print("Prediction failed")
            
    except Exception as e:
        print(f"Prediction error: {e}")


def evaluate_system(asr_system):
    print("Evaluating system...")
    
    if not asr_system.load_trained_models():
        print("Failed to load models")
        return
    
    try:
        results_file = asr_system.output_path / 'deep_learning_asr_results.json'
        if results_file.exists():
            import json
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print("Saved evaluation results:")
            asr_system._print_detailed_results(results)
            
        else:
            print("No saved results found")
            print("Run training first using --mode train")
            
    except Exception as e:
        print(f"Evaluation error: {e}")


def interactive_mode():
    print("\nInteractive Mode")
    print("Type 'exit' to quit")
    
    asr_system = ASRSystem()
    
    if not asr_system.load_trained_models():
        print("No trained models found")
        print("Train the system first")
        return
    
    while True:
        try:
            audio_file = input("\nEnter audio file path: ").strip()
            
            if audio_file.lower() == 'exit':
                print("Goodbye!")
                break
                
            if not audio_file:
                continue
                
            if not Path(audio_file).exists():
                print(f"File not found: {audio_file}")
                continue
            
            result = asr_system.predict_word(audio_file)
            
            if result:
                print(f"Result: {result['predicted_word']} ({result['confidence']*100:.1f}%)")
            else:
                print("Prediction failed")
                
        except KeyboardInterrupt:
            print("\nProgram stopped")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            print("No arguments provided, starting interactive mode...")
            print("For help: python main.py --help")
            
            choice = input("\nTrain system (t) or interactive mode (i)? [t/i]: ").strip().lower()
            
            if choice == 't':
                asr_system = ASRSystem()
                train_system(asr_system)
            elif choice == 'i':
                interactive_mode()
            else:
                print("Invalid choice")
        else:
            main()
            
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"General error: {e}")
        import traceback
        traceback.print_exc()