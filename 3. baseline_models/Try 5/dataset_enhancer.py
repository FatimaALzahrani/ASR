import os
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from audio_quality_enhancer import AudioQualityEnhancer

class DatasetEnhancer:
    def __init__(self, target_sr=22050):
        self.enhancer = AudioQualityEnhancer(target_sr=target_sr)
        
    def enhance_dataset(self, input_dir, output_dir, quality_analysis_file=None):
        print("Starting audio quality enhancement")
        print("="*40)
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        print(f"Input directory: {input_path}")
        print(f"Output directory: {output_path}")
        
        quality_data = None
        if quality_analysis_file and os.path.exists(quality_analysis_file):
            quality_data = pd.read_csv(quality_analysis_file)
            print(f"Quality analysis loaded from: {quality_analysis_file}")
        
        processed_count = 0
        error_count = 0
        processing_results = []
        
        for word_folder in input_path.iterdir():
            if word_folder.is_dir():
                word = word_folder.name
                output_word_folder = output_path / word
                output_word_folder.mkdir(parents=True, exist_ok=True)
                
                print(f"\nProcessing word: {word}")
                
                audio_files = list(word_folder.glob("*.wav"))
                
                for audio_file in audio_files:
                    try:
                        output_file = output_word_folder / audio_file.name
                        apply_noise_reduction = True
                        apply_normalization = True
                        apply_smart_trim = True
                        
                        if quality_data is not None:
                            file_info = quality_data[
                                (quality_data['filename'] == audio_file.name) & 
                                (quality_data['word'] == word)
                            ]
                            
                            if not file_info.empty:
                                file_data = file_info.iloc[0]
                                
                                if file_data['snr'] < 10:
                                    apply_noise_reduction = True
                                elif file_data['snr'] > 25:
                                    apply_noise_reduction = False
                                
                                if file_data['rms_mean'] > 0.1:
                                    apply_normalization = True
                                
                                if file_data['speech_ratio'] < 0.6:
                                    apply_smart_trim = True
                        
                        result = self.enhancer.enhance_audio_file(
                            str(audio_file), 
                            str(output_file),
                            apply_noise_reduction=apply_noise_reduction,
                            apply_normalization=apply_normalization,
                            apply_smart_trim=apply_smart_trim
                        )
                        
                        result['input_file'] = str(audio_file)
                        result['output_file'] = str(output_file)
                        result['word'] = word
                        result['filename'] = audio_file.name
                        
                        processing_results.append(result)
                        
                        if result['success']:
                            processed_count += 1
                            if processed_count % 200 == 0:
                                print(f"  Processed {processed_count} files...")
                        else:
                            error_count += 1
                            if error_count <= 5:
                                print(f"  Error in {audio_file.name}: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:
                            print(f"  Error processing {audio_file}: {e}")
        
        print(f"\nSuccessfully enhanced {processed_count} files")
        print(f"Failed to process {error_count} files")
        
        processing_df = pd.DataFrame(processing_results)
        processing_report_path = output_path / "processing_report.csv"
        processing_df.to_csv(processing_report_path, index=False, encoding='utf-8')
        
        print(f"Processing report saved to: {processing_report_path}")
        
        successful_results = processing_df[processing_df['success'] == True]
        
        if len(successful_results) > 0:
            print(f"\nEnhancement statistics:")
            print(f"  Average duration reduction: {(successful_results['original_duration'] - successful_results['final_duration']).mean():.3f} seconds")
            print(f"  Average RMS improvement: {(successful_results['final_rms'] - successful_results['original_rms']).mean():.4f}")
            
            trimmed_files = successful_results[successful_results['processing_steps'].str.contains('smart_trim', na=False)]
            if len(trimmed_files) > 0:
                print(f"  Files trimmed: {len(trimmed_files)}")
                print(f"  Average silence removed: {(trimmed_files['original_duration'] - trimmed_files['final_duration']).mean():.3f} seconds")
        
        return processing_results