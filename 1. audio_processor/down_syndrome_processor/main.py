from down_syndrome_audio_processor import DownSyndromeAudioProcessor


def main():
    print("Specialized Audio Processor for Down Syndrome Children")
    print("=" * 60)
    
    processor = DownSyndromeAudioProcessor()
    
    success = processor.process_all_files()
    
    if success:
        print("\nAudio enhancement completed successfully!")
        print("Enhanced data is ready for advanced training!")
        print("Next step: Train speech recognition model")
    else:
        print("\nAudio processing failed")


if __name__ == "__main__":
    main()