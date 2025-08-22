from speaker_manager import SpeakerManager
from audio_file_scanner import AudioFileScanner
from audio_analyzer import AudioAnalyzer
from statistics_calculator import StatisticsCalculator
from data_visualizer import DataVisualizer
from report_generator import ReportGenerator
from data_exporter import DataExporter


class ComprehensiveDataProcessor:
    def __init__(self, data_path='C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean'):
        self.data_path = data_path
        self.speaker_manager = SpeakerManager()
        self.file_scanner = AudioFileScanner(data_path)
        self.audio_analyzer = AudioAnalyzer()
    
    def run_comprehensive_analysis(self):
        print("Starting comprehensive data analysis...")
        print("=" * 60)
        
        valid_files = self.file_scanner.scan_files(self.speaker_manager, self.audio_analyzer)
        
        if valid_files == 0:
            print("No valid audio files found")
            return None
        
        stats_calc = StatisticsCalculator(self.file_scanner.audio_files)
        stats_calc.calculate_all_statistics()
        
        visualizer = DataVisualizer(stats_calc.df)
        visualizer.create_visualizations()
        
        report_gen = ReportGenerator(stats_calc.df, self.speaker_manager, stats_calc)
        report = report_gen.generate_detailed_report()
        
        exporter = DataExporter(stats_calc.df, self.speaker_manager, stats_calc.statistics)
        exporter.export_training_data()
        
        print("=" * 60)
        print("Comprehensive analysis completed successfully!")
        print(f"Analyzed {valid_files} audio files")
        print("Report saved: comprehensive_analysis_report.json")
        print("Visualizations saved: comprehensive_data_analysis.png")
        print("Training data exported: training_dataset.csv")
        
        return report