from data_processor import FinalDataProcessor
from data_splitter import DataSplitterFinal
from data_saver import DataSaverFinal


def main():
    print("Final Data Processing for Down Syndrome Speech Recognition")
    print("=" * 60)
    
    processor = FinalDataProcessor(min_samples=3)
    splitter = DataSplitterFinal()
    
    df = processor.process_data()
    
    if df.empty:
        print("Error: No data found")
        return
    
    general_splits, speaker_splits = splitter.create_splits(df)
    
    saver = DataSaverFinal(processor.processed_dir)
    saver.save_data(general_splits, speaker_splits)
    
    print("Data is ready for training!")


if __name__ == "__main__":
    main()