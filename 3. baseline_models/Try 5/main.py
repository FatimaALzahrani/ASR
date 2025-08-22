import argparse
import sys
from pathlib import Path

from dataset_enhancer import DatasetEnhancer
from enhanced_training_pipeline import EnhancedTrainingPipeline
from config import PathConfig

def enhance_audio(input_dir, output_dir, quality_file=None):
    print("Starting audio enhancement process...")
    
    enhancer = DatasetEnhancer()
    results = enhancer.enhance_dataset(input_dir, output_dir, quality_file)
    
    if results:
        print("Audio enhancement completed successfully!")
        return True
    else:
        print("Audio enhancement failed!")
        return False

def train_models(data_dir=None):
    print("Starting model training process...")
    
    if data_dir:
        # Update the data directory in the pipeline if needed
        pass
    
    pipeline = EnhancedTrainingPipeline()
    results = pipeline.run_training_comparison()
    
    if results:
        print("Model training completed successfully!")
        return True
    else:
        print("Model training failed!")
        return False

def run_full_pipeline(input_dir, output_dir, quality_file=None):
    print("Running full pipeline...")
    
    # Step 1: Enhance audio
    print("\nStep 1: Audio Enhancement")
    if not enhance_audio(input_dir, output_dir, quality_file):
        print("Pipeline failed at audio enhancement step")
        return False
    
    # Step 2: Train models
    print("\nStep 2: Model Training")
    if not train_models(output_dir):
        print("Pipeline failed at model training step")
        return False
    
    print("\nFull pipeline completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Audio Quality Enhancement and Training System')
    parser.add_argument('--mode', choices=['enhance', 'train', 'full'], required=True,
                        help='Operation mode: enhance audio, train models, or run full pipeline')
    parser.add_argument('--input-dir', default=PathConfig.DEFAULT_INPUT_DIR,
                        help='Input directory containing audio files')
    parser.add_argument('--output-dir', default=PathConfig.DEFAULT_OUTPUT_DIR,
                        help='Output directory for enhanced audio files')
    parser.add_argument('--quality-file', default=PathConfig.QUALITY_ANALYSIS_FILE,
                        help='Quality analysis file (optional)')
    
    args = parser.parse_args()
    
    # Validate paths
    if args.mode in ['enhance', 'full']:
        if not Path(args.input_dir).exists():
            print(f"Error: Input directory does not exist: {args.input_dir}")
            sys.exit(1)
    
    if args.mode == 'train':
        if not Path(args.output_dir).exists():
            print(f"Error: Enhanced data directory does not exist: {args.output_dir}")
            print("Please run audio enhancement first.")
            sys.exit(1)
    
    # Execute based on mode
    success = False
    
    if args.mode == 'enhance':
        success = enhance_audio(args.input_dir, args.output_dir, args.quality_file)
    
    elif args.mode == 'train':
        success = train_models(args.output_dir)
    
    elif args.mode == 'full':
        success = run_full_pipeline(args.input_dir, args.output_dir, args.quality_file)
    
    if success:
        print("\nOperation completed successfully!")
        sys.exit(0)
    else:
        print("\nOperation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()