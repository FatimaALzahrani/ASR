#!/usr/bin/env python3

from asr_system import ASRSystem

if __name__ == "__main__":
    print("Starting ASR System...")
    
    asr_system = ASRSystem()
    
    results = asr_system.run_evaluation()
    
    if results:
        print("ASR evaluation finished successfully!")
    else:
        print("Evaluation failed. Please check your data.")
