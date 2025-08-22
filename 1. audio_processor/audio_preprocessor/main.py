#!/usr/bin/env python3

from audio_preprocessor import FixedAudioPreprocessor


def main():
    print("Advanced Audio Data Processor for Children with Down Syndrome")
    print("=" * 70)
    
    processor = FixedAudioPreprocessor()
    processed_df = processor.run_complete_processing()
    
    print(f"\nSuccessfully processed {len(processed_df)} audio files with enhanced quality!")
    print("Now the processed data can be used for training with higher accuracy!")


if __name__ == "__main__":
    main()