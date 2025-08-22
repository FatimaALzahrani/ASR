import torch
import pandas as pd
import evaluate
from typing import Dict
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audio_processor import AudioProcessor
from config import Config

class WhisperEvaluator:
    def __init__(self, device: torch.device):
        self.device = device
        
        print(f"Loading model: {Config.WHISPER_MODEL_NAME}")
        
        self.processor = WhisperProcessor.from_pretrained(Config.WHISPER_MODEL_NAME)
        self.model = WhisperForConditionalGeneration.from_pretrained(Config.WHISPER_MODEL_NAME)
        self.model.to(self.device)
        
        self.wer_metric = evaluate.load("wer")
        self.audio_processor = AudioProcessor(self.processor, self.device)
        
    def evaluate_sample(self, data: pd.DataFrame, sample_name: str) -> Dict:
        print(f"Evaluating {sample_name}...")
        
        predictions = []
        references = []
        successful_files = 0
        failed_files = 0
        
        for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Evaluating {sample_name}"):
            file_path = row['file_path']
            reference_text = row['text']
            
            input_features = self.audio_processor.preprocess_audio(file_path)
            
            if input_features is None:
                failed_files += 1
                continue
            
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_features,
                        language=Config.LANGUAGE,
                        task=Config.TASK,
                        max_length=Config.MAX_LENGTH
                    )
                
                predicted_text = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0].strip()
                
                predictions.append(predicted_text)
                references.append(reference_text)
                successful_files += 1
                
            except Exception as e:
                print(f"Error predicting for file {file_path}: {e}")
                failed_files += 1
                continue
        
        if len(predictions) > 0:
            wer = self.wer_metric.compute(predictions=predictions, references=references)
            accuracy = 1 - wer
        else:
            wer = 1.0
            accuracy = 0.0
        
        results = {
            "sample_name": sample_name,
            "total_files": len(data),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "success_rate": successful_files / len(data) if len(data) > 0 else 0,
            "wer": wer,
            "accuracy": accuracy,
            "predictions_sample": predictions[:Config.PREDICTIONS_SAMPLE_SIZE],
            "references_sample": references[:Config.PREDICTIONS_SAMPLE_SIZE]
        }
        
        print(f"Results for {sample_name}:")
        print(f"   Successful files: {successful_files}/{len(data)}")
        print(f"   Word Error Rate (WER): {wer:.3f}")
        print(f"   Accuracy: {accuracy*100:.1f}%")
        
        return results