import os
import json
import pickle
import pandas as pd
from datetime import datetime


def ensure_directory_exists(directory_path):
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def get_file_list(directory, extensions=None):
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac']
    
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                file_list.append(os.path.join(root, file))
    
    return file_list


def save_json(data, file_path):
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {str(e)}")
        return False


def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {str(e)}")
        return None


def save_pickle(data, file_path):
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving pickle to {file_path}: {str(e)}")
        return False


def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle from {file_path}: {str(e)}")
        return None


def save_dataframe(df, file_path, format='csv'):
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        
        if format.lower() == 'csv':
            df.to_csv(file_path, index=False, encoding='utf-8')
        elif format.lower() == 'excel':
            df.to_excel(file_path, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return True
    except Exception as e:
        print(f"Error saving dataframe to {file_path}: {str(e)}")
        return False


def load_dataframe(file_path):
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"Error loading dataframe from {file_path}: {str(e)}")
        return None


def generate_timestamped_filename(base_name, extension, include_time=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S" if include_time else "%Y%m%d")
    return f"{base_name}_{timestamp}.{extension}"


def get_latest_file(directory, pattern):
    import glob
    
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    
    return max(files, key=os.path.getctime)


def organize_audio_files_by_speaker(input_dir, output_dir):
    ensure_directory_exists(output_dir)
    
    audio_files = get_file_list(input_dir)
    organized_files = {}
    
    for file_path in audio_files:
        filename = os.path.basename(file_path)
        
        parts = filename.split('_')
        if len(parts) >= 2:
            speaker = parts[0]
            word = parts[1].split('.')[0]
        else:
            speaker = 'unknown'
            word = os.path.splitext(filename)[0]
        
        speaker_dir = os.path.join(output_dir, speaker)
        ensure_directory_exists(speaker_dir)
        
        new_filename = f"{word}.wav"
        output_path = os.path.join(speaker_dir, new_filename)
        
        if speaker not in organized_files:
            organized_files[speaker] = []
        
        organized_files[speaker].append({
            'original_path': file_path,
            'new_path': output_path,
            'word': word
        })
    
    return organized_files


def create_file_manifest(directory, output_file):
    file_manifest = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)
            
            file_info = {
                'file_path': relative_path,
                'full_path': file_path,
                'size': os.path.getsize(file_path),
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }
            
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                from audio_utils import get_audio_duration, load_audio
                
                audio, sr = load_audio(file_path)
                if audio is not None:
                    file_info['duration'] = get_audio_duration(audio, sr)
                    file_info['sample_rate'] = sr
                    file_info['type'] = 'audio'
            
            file_manifest.append(file_info)
    
    save_json(file_manifest, output_file)
    return file_manifest


def cleanup_old_files(directory, days_old=30, dry_run=True):
    import time
    
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    deleted_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            if os.path.getmtime(file_path) < cutoff_time:
                if not dry_run:
                    try:
                        os.remove(file_path)
                        deleted_files.append(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {str(e)}")
                else:
                    deleted_files.append(file_path)
    
    if dry_run:
        print(f"Would delete {len(deleted_files)} files older than {days_old} days")
    else:
        print(f"Deleted {len(deleted_files)} files older than {days_old} days")
    
    return deleted_files


def validate_directory_structure(base_dir, required_subdirs):
    missing_dirs = []
    
    for subdir in required_subdirs:
        full_path = os.path.join(base_dir, subdir)
        if not os.path.exists(full_path):
            missing_dirs.append(subdir)
    
    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
        return False
    
    return True


def backup_directory(source_dir, backup_dir, compress=False):
    import shutil
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{timestamp}"
    
    if compress:
        backup_path = os.path.join(backup_dir, backup_name)
        shutil.make_archive(backup_path, 'zip', source_dir)
        return f"{backup_path}.zip"
    else:
        backup_path = os.path.join(backup_dir, backup_name)
        shutil.copytree(source_dir, backup_path)
        return backup_path