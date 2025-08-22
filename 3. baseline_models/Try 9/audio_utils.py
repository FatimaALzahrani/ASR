import os
import librosa
import soundfile as sf
import numpy as np
from settings import AUDIO_CONFIG


def load_audio(file_path, sr=None):
    if sr is None:
        sr = AUDIO_CONFIG['sample_rate']
    
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        return audio, sr
    except Exception as e:
        print(f"Error loading audio {file_path}: {str(e)}")
        return None, None


def save_audio(audio, file_path, sr=None):
    if sr is None:
        sr = AUDIO_CONFIG['sample_rate']
    
    try:
        sf.write(file_path, audio, sr)
        return True
    except Exception as e:
        print(f"Error saving audio {file_path}: {str(e)}")
        return False


def normalize_audio(audio):
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio))
    return audio


def trim_silence(audio, sr=None, top_db=20):
    if sr is None:
        sr = AUDIO_CONFIG['sample_rate']
    
    try:
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio
    except:
        return audio


def get_audio_duration(audio, sr=None):
    if sr is None:
        sr = AUDIO_CONFIG['sample_rate']
    
    return len(audio) / sr


def validate_audio_file(file_path):
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    if not file_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        return False, "Unsupported audio format"
    
    try:
        audio, sr = load_audio(file_path)
        if audio is None:
            return False, "Cannot load audio file"
        
        if len(audio) == 0:
            return False, "Empty audio file"
        
        duration = get_audio_duration(audio, sr)
        if duration < 0.1:
            return False, "Audio too short"
        
        if duration > 30:
            return False, "Audio too long"
        
        return True, "Valid audio file"
    
    except Exception as e:
        return False, f"Audio validation error: {str(e)}"


def preprocess_audio(audio, sr=None):
    if sr is None:
        sr = AUDIO_CONFIG['sample_rate']
    
    audio = normalize_audio(audio)
    audio = trim_silence(audio, sr)
    
    return audio


def batch_process_audio_files(input_dir, output_dir, process_func=preprocess_audio):
    os.makedirs(output_dir, exist_ok=True)
    processed_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                input_path = os.path.join(root, file)
                
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                audio, sr = load_audio(input_path)
                if audio is not None:
                    processed_audio = process_func(audio, sr)
                    
                    output_wav = os.path.splitext(output_path)[0] + '.wav'
                    if save_audio(processed_audio, output_wav, sr):
                        processed_files.append(output_wav)
                        print(f"Processed: {input_path} -> {output_wav}")
                    else:
                        print(f"Failed to save: {output_wav}")
                else:
                    print(f"Failed to load: {input_path}")
    
    return processed_files