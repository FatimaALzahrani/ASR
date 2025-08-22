import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from typing import Dict, List

class VisualizationGenerator:
    def __init__(self, output_path: str = "visualizations"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        plt.style.use('default')
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#7209B7'
        }
    
    def plot_model_comparison(self, results: Dict, save_path: str = None) -> str:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = list(results.keys())
        accuracies = [results[model].get('accuracy', 0) * 100 for model in models]
        
        bars = ax.bar(models, accuracies, color=self.colors['primary'], alpha=0.8)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('ASR Model Performance Comparison for Children with Down Syndrome', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Recognition Accuracy (%)', fontweight='bold')
        ax.set_ylim(0, 100)
        
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_path / 'model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison chart saved: {save_path}")
        return str(save_path)
    
    def plot_correction_effectiveness(self, before_scores: List[float], 
                                    after_scores: List[float], 
                                    save_path: str = None) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        x = np.arange(len(before_scores))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, before_scores, width, 
                       label='Before Correction', color=self.colors['secondary'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, after_scores, width,
                       label='After Correction', color=self.colors['primary'], alpha=0.8)
        
        ax1.set_xlabel('Cross-Validation Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Language Correction Effectiveness')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Fold {i+1}' for i in range(len(before_scores))])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        data_to_plot = [before_scores, after_scores]
        labels = ['Before Correction', 'After Correction']
        
        box_plot = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = [self.colors['secondary'], self.colors['primary']]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Statistical Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_path / 'correction_effectiveness.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Correction effectiveness chart saved: {save_path}")
        return str(save_path)
    
    def plot_training_curves(self, history_list: List[Dict], save_path: str = None) -> str:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for fold_idx, history in enumerate(history_list[:5]):
            if fold_idx < len(axes):
                ax = axes[fold_idx]
                ax.plot(history['accuracy'], label='Training', alpha=0.8)
                ax.plot(history['val_accuracy'], label='Validation', alpha=0.8)
                ax.set_title(f'Fold {fold_idx + 1}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        if len(axes) > len(history_list):
            for i in range(len(history_list), len(axes)):
                axes[i].set_visible(False)
        
        plt.suptitle('Training Curves Across Cross-Validation Folds', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_path / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved: {save_path}")
        return str(save_path)
    
    def create_summary_table(self, all_results: Dict, save_path: str = None) -> str:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        data = []
        for model_name, results in all_results.items():
            if isinstance(results, dict) and 'accuracy' in results:
                data.append([
                    model_name,
                    f"{results['accuracy']*100:.2f}%",
                    f"{results.get('precision', 0)*100:.2f}%",
                    f"{results.get('recall', 0)*100:.2f}%",
                    f"{results.get('f1_score', 0)*100:.2f}%"
                ])
        
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        table = ax.table(cellText=data, colLabels=headers,
                        cellLoc='center', loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        for i in range(len(headers)):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F8F9FA')
                else:
                    table[(i, j)].set_facecolor('white')
        
        ax.set_title('Performance Summary of ASR Models for Children with Down Syndrome', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_path / 'performance_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance summary table saved: {save_path}")
        return str(save_path)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: List[str], save_path: str = None) -> str:
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        if save_path is None:
            save_path = self.output_path / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved: {save_path}")
        return str(save_path)