import torch
from sklearn.metrics import accuracy_score


class ModelEvaluator:
    
    def __init__(self, device):
        self.device = device
    
    def evaluate_model(self, model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for data, target, _ in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                total_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(loader), correct / total
    
    def detailed_test(self, model, test_loader, model_name):
        model.eval()
        all_preds, all_targets, all_speakers = [], [], []
        
        with torch.no_grad():
            for data, target, speakers in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_speakers.extend(speakers)
        
        # Calculate overall accuracy
        accuracy = accuracy_score(all_targets, all_preds)
        
        # Calculate per-speaker accuracy
        speaker_analysis = {}
        for i, speaker in enumerate(all_speakers):
            if speaker not in speaker_analysis:
                speaker_analysis[speaker] = {'correct': 0, 'total': 0}
            
            speaker_analysis[speaker]['total'] += 1
            if all_preds[i] == all_targets[i]:
                speaker_analysis[speaker]['correct'] += 1
        
        # Print results
        print(f"Model {model_name} - Test Accuracy: {accuracy:.3f}")
        for speaker, stats in speaker_analysis.items():
            acc = stats['correct'] / stats['total']
            print(f"  {speaker}: {acc:.3f} ({stats['correct']}/{stats['total']})")
        
        return accuracy, {
            'accuracy': accuracy,
            'speaker_analysis': speaker_analysis,
            'predictions': all_preds,
            'targets': all_targets
        }