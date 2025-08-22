import numpy as np
from typing import Dict, List, Tuple, Generator
from audio_processor import AdvancedAudioProcessor


class EfficientDataLoader:
    def __init__(self, file_paths: List[str], transcripts: List[str], speaker_ids: List[int], batch_size: int = 16):
        self.file_paths = file_paths
        self.transcripts = transcripts
        self.speaker_ids = speaker_ids
        self.batch_size = batch_size
        self.audio_processor = AdvancedAudioProcessor()
        
    def __len__(self):
        return len(self.file_paths)
    
    def load_batch(self, start_idx: int, end_idx: int) -> Tuple[List[np.ndarray], List[str], List[int]]:
        batch_audio = []
        batch_transcripts = []
        batch_speaker_ids = []
        
        for i in range(start_idx, min(end_idx, len(self.file_paths))):
            try:
                audio = self.audio_processor.load_and_preprocess(self.file_paths[i])
                if len(audio) > 0 and not np.all(audio == 0):
                    batch_audio.append(audio)
                    batch_transcripts.append(self.transcripts[i])
                    batch_speaker_ids.append(self.speaker_ids[i])
            except Exception as e:
                print(f"Error loading {self.file_paths[i]}: {e}")
                continue
                
        return batch_audio, batch_transcripts, batch_speaker_ids
    
    def get_batches(self) -> Generator[Tuple[List[np.ndarray], List[str], List[int]], None, None]:
        for i in range(0, len(self.file_paths), self.batch_size):
            yield self.load_batch(i, i + self.batch_size)
    
    def sample_subset(self, num_samples: int) -> Tuple[List[np.ndarray], List[str], List[int]]:
        indices = np.random.choice(len(self.file_paths), size=min(num_samples, len(self.file_paths)), replace=False)
        
        audio_list = []
        transcript_list = []
        speaker_list = []
        
        for idx in indices:
            try:
                audio = self.audio_processor.load_and_preprocess(self.file_paths[idx])
                if len(audio) > 0 and not np.all(audio == 0):
                    audio_list.append(audio)
                    transcript_list.append(self.transcripts[idx])
                    speaker_list.append(self.speaker_ids[idx])
            except Exception as e:
                print(f"Error in subset sampling: {e}")
                continue
                
        return audio_list, transcript_list, speaker_list