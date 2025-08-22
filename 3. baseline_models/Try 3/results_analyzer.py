import numpy as np
import matplotlib.pyplot as plt


class ResultsAnalyzer:
    def __init__(self):
        pass
        
    def create_comprehensive_plots(self, all_results):
        print("Creating comprehensive plots...")
        
        models = []
        cv_means = []
        cv_stds = []
        test_accs = []
        overfitting_gaps = []
        f1_scores = []
        
        for name, result in all_results.items():
            if result is not None:
                models.append(name.replace(' (Regularized)', '').replace(' (High Regularization)', ''))
                cv_means.append(result['cv_mean'])
                cv_stds.append(result['cv_std'])
                test_accs.append(result['test_accuracy'])
                overfitting_gaps.append(result['overfitting_gap'])
                f1_scores.append(result['f1_score'])
        
        if not models:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].bar(range(len(models)), cv_means, yerr=cv_stds, 
                      alpha=0.7, capsize=5, color='skyblue')
        axes[0, 0].set_title('Cross-Validation Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        
        axes[0, 1].bar(range(len(models)), test_accs, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Test Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        
        colors = ['red' if gap > 0.2 else 'orange' if gap > 0.1 else 'green' 
                 for gap in overfitting_gaps]
        axes[0, 2].bar(range(len(models)), overfitting_gaps, 
                      alpha=0.7, color=colors)
        axes[0, 2].set_title('Overfitting Gap')
        axes[0, 2].set_ylabel('Training - Test Accuracy')
        axes[0, 2].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7)
        axes[0, 2].axhline(y=0.2, color='red', linestyle='--', alpha=0.7)
        axes[0, 2].set_xticks(range(len(models)))
        axes[0, 2].set_xticklabels(models, rotation=45, ha='right')
        
        axes[1, 0].bar(range(len(models)), f1_scores, alpha=0.7, color='gold')
        axes[1, 0].set_title('F1-Score')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_xticks(range(len(models)))
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[1, 1].bar(x - width, cv_means, width, label='CV Mean', alpha=0.7)
        axes[1, 1].bar(x, test_accs, width, label='Test Accuracy', alpha=0.7)
        axes[1, 1].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.7)
        
        axes[1, 1].set_title('Comprehensive Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].legend()
        
        axes[1, 2].text(0.1, 0.9, 'Best Models:', fontsize=14, fontweight='bold')
        
        sorted_indices = np.argsort(test_accs)[::-1]
        y_pos = 0.8
        
        for i, idx in enumerate(sorted_indices[:5]):
            model_name = models[idx]
            test_acc = test_accs[idx]
            cv_mean = cv_means[idx]
            f1 = f1_scores[idx]
            
            text = f"{i+1}. {model_name}\n   Test: {test_acc:.3f}, CV: {cv_mean:.3f}, F1: {f1:.3f}"
            axes[1, 2].text(0.1, y_pos, text, fontsize=9)
            y_pos -= 0.15
        
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('improved_models_comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        print("Plots saved to: improved_models_comprehensive_analysis.png")