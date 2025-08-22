import warnings
warnings.filterwarnings('ignore')

from main_processor import ComprehensiveDataProcessor


def main():
    print("Comprehensive Audio Data Processor for Children with Down Syndrome")
    print("=" * 60)
    
    processor = ComprehensiveDataProcessor()
    report = processor.run_comprehensive_analysis()
    
    if report:
        print("\nResults Summary:")
        print(f"• Total files: {report['dataset_overview']['total_files']}")
        print(f"• Total words: {report['dataset_overview']['total_words']}")
        print(f"• Total speakers: {report['dataset_overview']['total_speakers']}")
        print(f"• Total duration: {report['dataset_overview']['total_duration_hours']} hours")
        print(f"• Average file duration: {report['dataset_overview']['average_file_duration']} seconds")
        
        print("\nTop speakers by quality:")
        speaker_scores = [(speaker, data['average_quality']) 
                         for speaker, data in report['speaker_analysis'].items()]
        speaker_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (speaker, score) in enumerate(speaker_scores[:3], 1):
            print(f"{i}. {speaker}: {score:.3f}")


if __name__ == "__main__":
    main()