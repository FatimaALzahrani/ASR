import os
import argparse
import logging
from asr_system import ProfessionalASRSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Professional Arabic Speech Recognition System')
    parser.add_argument('--data_path', required=True, help='Path to audio data folder')
    parser.add_argument('--output_dir', default='professional_asr_models', help='Output directory for models')
    parser.add_argument('--min_samples', type=int, default=3, help='Minimum samples per word')
    parser.add_argument('--test_audio', help='Path to test audio file')
    parser.add_argument('--speaker', help='Speaker name for test audio')
    
    args = parser.parse_args()
    
    asr_system = ProfessionalASRSystem()
    
    logger.info("Training professional ASR system")
    training_results = asr_system.train(
        audio_data_path=args.data_path,
        min_samples_per_word=args.min_samples
    )
    
    asr_system.save_system(args.output_dir)
    
    if args.test_audio and os.path.exists(args.test_audio):
        logger.info(f"Testing with audio: {args.test_audio}")
        result = asr_system.recognize_speech(args.test_audio, args.speaker)
        
        print("\\nRecognition Result:")
        print(f"Acoustic Prediction: {result['acoustic_prediction']}")
        print(f"Acoustic Confidence: {result['acoustic_confidence']:.4f}")
        print(f"Final Prediction: {result['final_prediction']}")
        print(f"Final Confidence: {result['final_confidence']:.4f}")
        print(f"Corrections Applied: {result['corrections_applied']}")
    
    logger.info("Professional ASR system training and saving completed")
    
    return training_results

if __name__ == "__main__":
    main()
