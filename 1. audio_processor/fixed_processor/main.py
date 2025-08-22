#!/usr/bin/env python3

from down_syndrome_processor import DownSyndromeProcessor


def main():
    print("Enhanced Down Syndrome Children Speech Data Processor")
    print("=" * 70)
    
    processor = DownSyndromeProcessor()
    
    success = processor.run_complete_analysis()
    
    if success:
        print("\nData is completely ready for the next step!")
    else:
        print("\nData processing failed. Please review the errors above.")


if __name__ == "__main__":
    main()