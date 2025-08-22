import json
import torch
import numpy as np
from pathlib import Path


def save_model(model, filepath):
    """Save model state dict"""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):
    """Load model state dict"""
    if Path(filepath).exists():
        model.load_state_dict(torch.load(filepath, map_location=device))
        print(f"Model loaded from {filepath}")
        return True
    else:
        print(f"Model file not found: {filepath}")
        return False


def save_results(results, filepath):
    """Save training results to JSON"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results: {e}")


def print_model_summary(model, input_shape):
    """Print model parameter summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Input shape: {input_shape}")


def calculate_accuracy(predictions, targets):
    """Calculate accuracy from predictions and targets"""
    correct = (predictions == targets).sum().item()
    total = len(targets)
    return correct / total


def create_speaker_analysis(predictions, targets, speakers):
    """Create per-speaker accuracy analysis"""
    speaker_results = {}
    
    for speaker in set(speakers):
        speaker_mask = [s == speaker for s in speakers]
        speaker_preds = [predictions[i] for i, mask in enumerate(speaker_mask) if mask]
        speaker_targets = [targets[i] for i, mask in enumerate(speaker_mask) if mask]
        
        if speaker_preds:
            accuracy = calculate_accuracy(np.array(speaker_preds), np.array(speaker_targets))
            speaker_results[speaker] = {
                'accuracy': accuracy,
                'samples': len(speaker_preds)
            }
    
    return speaker_results


def format_time(seconds):
    """Format training time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "data/processed/train.csv",
        "data/processed/validation.csv",
        "data/processed/test.csv"
    ]
    
    missing_files = []
    for filepath in required_files:
        if not Path(filepath).exists():
            missing_files.append(filepath)
    
    if missing_files:
        print("Missing required data files:")
        for filepath in missing_files:
            print(f"  {filepath}")
        return False
    
    return True


def setup_directories():
    """Create necessary output directories"""
    directories = ['models', 'logs', 'results']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("Output directories created")


def log_training_info(model_name, epoch, train_acc, val_acc, loss):
    """Log training information"""
    log_entry = {
        'model': model_name,
        'epoch': epoch,
        'train_accuracy': train_acc,
        'validation_accuracy': val_acc,
        'loss': loss
    }
    
    log_file = Path('logs') / f'{model_name}_training.log'
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')