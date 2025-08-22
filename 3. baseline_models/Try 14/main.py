import argparse
import sys
from pathlib import Path
from speaker_analyzer import SpeakerAnalyzer

def main():
    parser = argparse.ArgumentParser(
        description="Down Syndrome Speech Recognition System - Advanced Speaker-Adaptive ASR"
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        default="C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean",
        help="Path to audio data directory (default: C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean)"
    )
    
    parser.add_argument(
        '--output-path', 
        type=str, 
        default="output_files",
        help="Path to save results (default: results)"
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path '{data_path}' does not exist!")
        sys.exit(1)
    
    print("="*70)
    print("Down Syndrome Speech Recognition System")
    print("Advanced Speaker-Adaptive ASR")
    print("="*70)
    print(f"Data path: {data_path}")
    print(f"Output path: {args.output_path}")
    print("="*70)
    
    try:
        analyzer = SpeakerAnalyzer(
            data_path=str(data_path), 
            output_path=args.output_path
        )
        
        results = analyzer.run_analysis()
        
        if results:
            print(f"\n{'='*70}")
            print(f"SUCCESS: Analysis completed!")
            print(f"Results saved to: {args.output_path}/")
            print(f"Models saved to: {args.output_path}/models.pkl")
            print(f"Report saved to: {args.output_path}/results.json")
            print(f"{'='*70}")
            
            total_speakers = len(results)
            avg_accuracy = sum(r['best_accuracy'] for r in results.values()) / total_speakers
            
            print(f"\nQuick Summary:")
            print(f"  Trained speakers: {total_speakers}")
            print(f"  Average accuracy: {avg_accuracy*100:.2f}%")
            print(f"  System status: Ready for deployment")
            
        else:
            print(f"\nERROR: No results obtained!")
            print(f"Please check:")
            print(f"  1. Audio files exist in {data_path}")
            print(f"  2. Files are in .wav format")
            print(f"  3. Folder structure is correct")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
