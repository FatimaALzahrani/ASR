import argparse
import warnings
import sys
from pathlib import Path

from data_processor import DownSyndromeDataProcessor
from data_splitter import DataSplitter
import config

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process Down Syndrome speech recognition data'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default=config.DATA_ROOT,
        help=f'Root directory containing audio data (default: {config.DATA_ROOT})'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=config.TEST_SIZE,
        help=f'Test set size ratio (default: {config.TEST_SIZE})'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=config.VALIDATION_SIZE,
        help=f'Validation set size ratio (default: {config.VALIDATION_SIZE})'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=2,
        help='Minimum samples per word to include (default: 2)'
    )
    
    parser.add_argument(
        '--skip-visualizations',
        action='store_true',
        help='Skip generating visualization plots'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip interactive validation prompts'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def validate_data_structure(data_root: Path) -> bool:
    clean_audio_dir = data_root / config.CLEAN_AUDIO_DIR
    
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return False
    
    if not clean_audio_dir.exists():
        print(f"Error: Clean audio directory not found: {clean_audio_dir}")
        print(f"Expected structure: {data_root}/{config.CLEAN_AUDIO_DIR}/[word_folders]/[audio_files.wav]")
        return False
    
    # Check for word folders
    word_folders = [d for d in clean_audio_dir.iterdir() if d.is_dir()]
    if not word_folders:
        print(f"Error: No word folders found in: {clean_audio_dir}")
        return False
    
    # Check for audio files
    total_audio_files = 0
    for word_folder in word_folders:
        audio_files = list(word_folder.glob("*.wav"))
        total_audio_files += len(audio_files)
    
    if total_audio_files == 0:
        print(f"Error: No .wav audio files found in word folders")
        return False
    
    print(f"Data structure validation passed:")
    print(f"  Data root: {data_root}")
    print(f"  Word folders: {len(word_folders)}")
    print(f"  Audio files: {total_audio_files}")
    
    return True


def main():
    args = parse_arguments()
    
    print("Down Syndrome Speech Recognition Data Processor")
    print("=" * 55)
    print(f"Data root: {args.data_root}")
    print(f"Test size: {args.test_size}")
    print(f"Validation size: {args.val_size}")
    print("=" * 55)
    
    try:
        # Validate data structure
        data_root = Path(args.data_root)
        if not validate_data_structure(data_root):
            sys.exit(1)
        
        # Initialize processor
        processor = DownSyndromeDataProcessor(args.data_root, args.skip_validation)
        
        # Update splitter settings if provided
        if args.test_size != config.TEST_SIZE or args.val_size != config.VALIDATION_SIZE:
            processor.splitter = DataSplitter(args.test_size, args.val_size)
        
        # Run complete analysis
        success = processor.run_complete_analysis()
        
        if success:
            # Print summary statistics
            stats = processor.get_summary_stats()
            if stats and args.verbose:
                print("\nDetailed Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            print("\nData processing completed successfully!")
            print("Ready for next step: Audio enhancement and model training")
            
        else:
            print("Data processing failed. Check error messages above.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()