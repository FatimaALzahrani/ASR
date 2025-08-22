#python main.py --data_path "C:\Users\فاطمة الزهراني\Desktop\ابحاث\الداون\Data\clean" --output_dir "output_files"

from config import *
from speaker_asr_system import SpeakerASRSystem

def main():
    parser = argparse.ArgumentParser(description='Speaker-Specific ASR System')
    parser.add_argument('--data_path', required=True, help='Path to data folder')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    parser.add_argument('--min_samples', type=int, default=Config.MIN_SAMPLES_PER_WORD, help='Minimum samples per word')
    parser.add_argument('--random_seed', type=int, default=Config.RANDOM_STATE, help='Random seed')
    
    args = parser.parse_args()
    
    asr_system = SpeakerASRSystem(random_state=args.random_seed)
    
    results = asr_system.run_complete_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    return results

if __name__ == "__main__":
    main()