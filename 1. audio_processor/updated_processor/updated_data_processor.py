import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from speaker_mapper import SpeakerMapper
from audio_structure_scanner import AudioStructureScanner
from dataset_splitter import DatasetSplitter
from mapping_creator import MappingCreator
from statistics_computer import StatisticsComputer
from data_saver import DataSaver


class UpdatedDataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.audio_root = self.data_dir / "C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean"
        self.processed_dir = self.data_dir / "C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Codes/Clean/1. audio_processor/updated_data_processor/output_files/processed"
        
        self.processed_dir.mkdir(exist_ok=True)
        
        self.target_sample_rate = 16000
        self.max_duration = 30.0
        self.min_duration = 0.5
        
        self.speaker_mapper = SpeakerMapper()
        self.audio_scanner = AudioStructureScanner(
            self.audio_root, self.min_duration, self.max_duration
        )
        self.dataset_splitter = DatasetSplitter()
        self.mapping_creator = MappingCreator()
        self.statistics_computer = StatisticsComputer()
        self.data_saver = DataSaver(self.processed_dir, self.data_dir)
    
    def process_all(self) -> bool:
        print("Starting data processing with new structure...")
        
        data = self.audio_scanner.scan_audio_structure(self.speaker_mapper)
        if data.empty:
            print("No valid data to process")
            return False
        
        splits = self.dataset_splitter.create_dataset_splits(data)
        mappings = self.mapping_creator.create_mappings(data)
        statistics = self.statistics_computer.calculate_statistics(splits)
        
        self.data_saver.save_processed_data(splits, mappings, statistics)
        
        print("Data processing completed successfully!")
        return True