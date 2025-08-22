import argparse
import warnings
from standalone_asr_system import StandaloneASRSystem

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='Standalone Advanced Arabic Speech Recognition System')
    parser.add_argument('--data_path', required=True, help='Path to data folder')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    parser.add_argument('--min_samples', type=int, default=3, help='Minimum samples per word')
    parser.add_argument('--min_quality', type=float, default=0.3, help='Minimum quality score')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    asr_system = StandaloneASRSystem(
        random_state=args.random_seed,
        min_samples_per_word=args.min_samples,
        min_quality_score=args.min_quality
    )
    
    results = asr_system.run_complete_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    print("Standalone Advanced ASR System completed successfully")
    print(f"Results saved to: {args.output_dir}")
    print(f"CSV files: {len(results['csv_files'])} files created")
    print(f"Report: {results['report_path']}")
    print(f"Summary: {results['summary_path']}")


if __name__ == "__main__":
    main()
