import os
import sys
import argparse
from datetime import datetime
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from train_models import ModelTrainer
from integrated_system import IntegratedSpeechSystem
from file_utils import ensure_directory_exists, validate_directory_structure
from settings import FEATURES_PATH, RESULTS_PATH, PROCESSED_DATA_PATH


def setup_directories():
    required_dirs = [
        FEATURES_PATH,
        RESULTS_PATH,
        f"{RESULTS_PATH}/acoustic_models",
        f"{RESULTS_PATH}/language_models", 
        f"{RESULTS_PATH}/correction_models",
        f"{RESULTS_PATH}/ensemble_models",
        f"{RESULTS_PATH}/reports"
    ]
    
    for directory in required_dirs:
        ensure_directory_exists(directory)
    
    print("Directory structure validated")


def train_models(args):
    print("Starting model training...")
    
    setup_directories()
    
    trainer = ModelTrainer(
        features_path=args.features_path,
        results_path=args.results_path
    )
    
    models_to_train = args.models if args.models else ['acoustic', 'language', 'corrector', 'ensemble']
    if 'all' in models_to_train:
        models_to_train = ['acoustic', 'language', 'corrector', 'ensemble']
    
    success = trainer.run_complete_training(models_to_train)
    
    if success:
        print("\nModel training completed successfully!")
        return 0
    else:
        print("\nModel training failed!")
        return 1


def recognize_speech(args):
    print("Starting speech recognition...")
    
    system = IntegratedSpeechSystem(models_path=args.models_path)
    
    if not system.load_models():
        print("Failed to load models. Please train models first.")
        return 1
    
    if args.audio_file:
        if not os.path.exists(args.audio_file):
            print(f"Audio file not found: {args.audio_file}")
            return 1
        
        result = system.recognize_speech(args.audio_file, args.speaker)
        
        if result['success']:
            print(f"\nRecognition Result:")
            print(f"Recognized Word: {result['final_word']}")
            print(f"Confidence: {result['correction_confidence']:.3f}")
            print(f"Processing Method: {result['correction_method']}")
            
            if result['correction_applied']:
                print(f"Original Prediction: {result['acoustic_prediction']}")
                print(f"After Correction: {result['final_word']}")
            
            if args.speaker:
                speaker_info = result.get('speaker_info', {})
                if speaker_info:
                    print(f"Speaker Info: {speaker_info}")
            
            return 0
        else:
            print(f"Recognition failed: {result['error']}")
            return 1
    
    elif args.audio_dir:
        if not os.path.exists(args.audio_dir):
            print(f"Audio directory not found: {args.audio_dir}")
            return 1
        
        audio_files = []
        for file in os.listdir(args.audio_dir):
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(args.audio_dir, file))
        
        if not audio_files:
            print("No audio files found in directory")
            return 1
        
        audio_files.sort()
        result = system.recognize_speech_sequence(audio_files, args.speaker)
        
        print(f"\nSequence Recognition Result:")
        print(f"Recognized Text: '{result['recognized_text']}'")
        print(f"Success Rate: {result['success_rate']:.1%}")
        print(f"Files Processed: {result['successful_recognitions']}/{result['total_files']}")
        
        return 0
    
    else:
        print("Please provide --audio-file or --audio-dir")
        return 1


def test_system(args):
    print("Testing integrated system...")
    
    system = IntegratedSpeechSystem(models_path=args.models_path)
    
    if not system.load_models():
        print("Failed to load models. Please train models first.")
        return 1
    
    success = system.test_system_with_sample()
    
    if success:
        print("System test passed!")
        return 0
    else:
        print("System test failed!")
        return 1


def show_status(args):
    print("Speech Recognition System Status")
    print("=" * 50)
    
    print(f"Features Path: {args.features_path}")
    print(f"Results Path: {args.results_path}")
    print(f"Models Path: {args.models_path}")
    
    print(f"\nDirectory Status:")
    dirs_to_check = [
        args.features_path,
        args.results_path,
        f"{args.results_path}/acoustic_models",
        f"{args.results_path}/language_models",
        f"{args.results_path}/correction_models"
    ]
    
    for directory in dirs_to_check:
        exists = "✓" if os.path.exists(directory) else "✗"
        print(f"  {exists} {directory}")
    
    features_file = os.path.join(args.features_path, "features_for_modeling.csv")
    features_exists = "✓" if os.path.exists(features_file) else "✗"
    print(f"  {features_exists} Features file: {features_file}")
    
    model_files = [
        ("Acoustic Model", f"{args.results_path}/acoustic_models/acoustic_model_*.pkl"),
        ("Language Model", f"{args.results_path}/language_models/language_model_*.pkl"), 
        ("Corrector", f"{args.results_path}/correction_models/corrector_*.pkl")
    ]
    
    print(f"\nModel Status:")
    for model_name, pattern in model_files:
        import glob
        files = glob.glob(pattern)
        status = "✓" if files else "✗"
        count = len(files)
        print(f"  {status} {model_name}: {count} file(s)")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Speech Recognition System for Children with Down Syndrome',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models
  python main.py train
  
  # Train specific models
  python main.py train --models acoustic language
  
  # Recognize single file
  python main.py recognize --audio-file path/to/audio.wav --speaker "أحمد"
  
  # Recognize directory of files
  python main.py recognize --audio-dir path/to/directory --speaker "هيفاء"
  
  # Test system
  python main.py test
  
  # Check status
  python main.py status
        """
    )
    
    parser.add_argument('--features-path', type=str, default=FEATURES_PATH,
                       help='Path to features directory')
    parser.add_argument('--results-path', type=str, default=RESULTS_PATH,
                       help='Path to results directory')
    parser.add_argument('--models-path', type=str, default=RESULTS_PATH,
                       help='Path to trained models directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--models', nargs='+', 
                             choices=['acoustic', 'language', 'corrector', 'ensemble', 'all'],
                             help='Models to train')
    
    recognize_parser = subparsers.add_parser('recognize', help='Recognize speech')
    recognize_group = recognize_parser.add_mutually_exclusive_group(required=True)
    recognize_group.add_argument('--audio-file', type=str, help='Single audio file to recognize')
    recognize_group.add_argument('--audio-dir', type=str, help='Directory of audio files to recognize')
    recognize_parser.add_argument('--speaker', type=str, help='Speaker name')
    
    test_parser = subparsers.add_parser('test', help='Test integrated system')
    
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    print("Speech Recognition System for Children with Down Syndrome")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if args.command == 'train':
        return train_models(args)
    elif args.command == 'recognize':
        return recognize_speech(args)
    elif args.command == 'test':
        return test_system(args)
    elif args.command == 'status':
        return show_status(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())