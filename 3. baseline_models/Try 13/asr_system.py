#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from config import Config
from data_loader import DataLoader
from model_trainer import ModelTrainer
from results_analyzer import ResultsAnalyzer

class AdvancedASRSystem:
    
    def __init__(self, data_path=None, output_path=None):
        self.data_path = Path(data_path) if data_path else Path(Config.DATA_PATH)
        self.output_path = Path(output_path) if output_path else Path(Config.OUTPUT_PATH)
        self.output_path.mkdir(exist_ok=True)
        
        self.speaker_profiles = Config.SPEAKER_PROFILES
        self.word_categories = Config.WORD_CATEGORIES
        self.difficulty_levels = Config.DIFFICULTY_LEVELS
        self.word_quality_map = Config.WORD_QUALITY_MAP
        
        self.speaker_models = {}
        self.speaker_scalers = {}
        self.speaker_results = {}
        self.speaker_data = {}
        self.global_stats = {}
        
        self.data_loader = DataLoader(
            self.data_path, 
            self.speaker_profiles, 
            self.word_categories, 
            self.difficulty_levels, 
            self.word_quality_map
        )
        
        self.model_trainer = ModelTrainer()
        self.results_analyzer = ResultsAnalyzer(self.output_path)
    
    def run_comprehensive_analysis(self):
        print("Advanced Down Syndrome Children Speech Recognition System")
        print("Comprehensive database analysis with advanced optimizations")
        print("="*90)
        
        speaker_data = self.data_loader.load_comprehensive_data()
        
        if not speaker_data:
            print("No data loaded!")
            return None
        
        print(f"\nOptimized data summary:")
        total_samples = 0
        total_words = set()
        
        for speaker, df in speaker_data.items():
            samples = len(df)
            words = len(df['word'].unique())
            total_samples += samples
            total_words.update(df['word'].unique())
            
            print(f"   {speaker}: {samples:,} samples, {words} words")
        
        print(f"   Total: {total_samples:,} samples, {len(total_words)} unique words")
        
        print(f"\nTraining advanced models:")
        all_results = {}
        
        for speaker, df in speaker_data.items():
            result = self.model_trainer.train_advanced_model(speaker, df)
            if result:
                all_results[speaker] = result
                
                self.speaker_models[speaker] = {
                    'model': result['model'],
                    'model_name': result['best_model'],
                    'results': result['results'],
                    'feature_count': result['feature_count'],
                    'word_count': result['word_count'],
                    'sample_count': result['sample_count'],
                    'profile': result['profile'],
                    'words': result['word_list']
                }
                
                self.speaker_scalers[speaker] = result['scalers']
        
        self.speaker_data = speaker_data
        self.global_stats = self.data_loader.global_stats
        
        self.results_analyzer.save_comprehensive_results(
            all_results, self.speaker_models, self.speaker_scalers,
            self.speaker_profiles, self.word_categories, 
            self.difficulty_levels, self.global_stats
        )
        
        self.results_analyzer.print_comprehensive_report(all_results)
        
        self.results_analyzer.perform_additional_analysis(
            all_results, self.speaker_profiles, 
            self.word_categories, self.difficulty_levels
        )
        
        return all_results
