from asr_system import ASRSystem


if __name__ == "__main__":
    print("Starting Ultimate ASR System...")
    
    asr_system = ASRSystem()
    
    results = asr_system.run_ultimate_evaluation()
    
    if results:
        print(f"\nEVALUATION COMPLETED!")
        print(f"Maximum Accuracy: {results['absolute_best']['accuracy']*100:.2f}%")
        print(f"Words Processed: {results['dataset_info']['words']}")
        print(f"Results saved to: {asr_system.output_path}")
    else:
        print("Evaluation failed!")