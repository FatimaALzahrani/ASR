from updated_data_processor import UpdatedDataProcessor


def main():
    print("Updated Data Processor for Children with Down Syndrome Speech Recognition")
    print("=" * 70)
    
    processor = UpdatedDataProcessor()
    success = processor.process_all()
    
    if success:
        print("\nData is ready for training!")
    else:
        print("\nData processing failed. Please check files and directories.")


if __name__ == "__main__":
    main()