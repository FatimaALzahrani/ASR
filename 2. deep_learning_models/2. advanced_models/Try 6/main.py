import os
import sys
import warnings
import tensorflow as tf

from asr_system import ComprehensiveASRSystem

warnings.filterwarnings('ignore')
tf.random.set_seed(42)

def setup_environment():
    tf.random.set_seed(42)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU acceleration enabled: {len(gpus)} GPU(s) found")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("Running on CPU")

def main():
    print("Comprehensive ASR System for Children with Down Syndrome")
    print("Academic Research Implementation")
    print("=" * 70)
    
    setup_environment()
    
    default_data_path = input("Enter data path (or press Enter for default): ").strip()
    if not default_data_path:
        default_data_path = "C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Data/clean"
    
    output_path = input("Enter output path (or press Enter for default): ").strip()
    if not output_path:
        output_path = "comprehensive_asr_results"
    
    if not os.path.exists(default_data_path):
        print(f"Error: Data path '{default_data_path}' does not exist")
        print("Please provide a valid data path")
        return 1
    
    print(f"Data path: {default_data_path}")
    print(f"Output path: {output_path}")
    print()
    
    try:
        asr_system = ComprehensiveASRSystem(default_data_path, output_path)
        results = asr_system.run_complete_evaluation()
        
        if "error" not in results:
            print("\n" + "=" * 70)
            print("FINAL RESULTS SUMMARY")
            print("=" * 70)
            
            if 'neural_results' in results:
                print("\nDeep Learning Models Performance:")
                neural_results = results['neural_results']
                sorted_models = sorted(neural_results.items(), 
                                     key=lambda x: x[1].get('accuracy', 0), reverse=True)
                
                for model_name, result in sorted_models:
                    accuracy = result.get('accuracy', 0)
                    std = result.get('std', 0)
                    print(f"  {model_name:20}: {accuracy:.4f} ± {std:.4f}")
                
                best_model, best_result = sorted_models[0]
                print(f"\nBest Performing Model: {best_model}")
                print(f"Best Accuracy: {best_result['accuracy']:.4f}")
            
            if 'baseline_results' in results:
                print("\nBaseline Models Performance:")
                baseline_results = results['baseline_results']
                baseline_accs = [(name, res.get('accuracy', 0)) 
                               for name, res in baseline_results.items() 
                               if isinstance(res, dict)]
                baseline_accs.sort(key=lambda x: x[1], reverse=True)
                
                for name, acc in baseline_accs:
                    print(f"  {name:20}: {acc:.4f}")
            
            if 'correction_results' in results:
                correction = results['correction_results']
                print(f"\nLanguage Correction System:")
                print(f"  Original Accuracy:     {correction['original_accuracy']:.4f}")
                print(f"  With Simulated Errors: {correction['simulated_accuracy']:.4f}")
                print(f"  After Correction:      {correction['corrected_accuracy']:.4f}")
                print(f"  Improvement:          +{correction['improvement']:.4f}")
            
            if 'statistical_analysis' in results:
                stats = results['statistical_analysis']
                print(f"\nStatistical Analysis:")
                
                if 'correction_effectiveness' in stats:
                    corr_stats = stats['correction_effectiveness']
                    print(f"  Correction p-value:    {corr_stats['p_value']:.6f}")
                    print(f"  Effect size (Cohen's d): {corr_stats['effect_size_cohens_d']:.4f}")
                    print(f"  Statistically significant: {corr_stats['significant']}")
                
                if 'method_comparison' in stats:
                    comp = stats['method_comparison']
                    print(f"  Improvement over baseline: {comp['relative_improvement']:.1f}%")
            
            print(f"\nAll results and visualizations saved to: {output_path}")
            
            if 'visualization_paths' in results:
                print("\nGenerated visualizations:")
                for viz_name, viz_path in results['visualization_paths'].items():
                    print(f"  - {viz_name}")
            
            print("\n" + "=" * 70)
            print("Evaluation completed successfully!")
            print("Ready for academic publication and analysis.")
            
            print("\nKey Academic Metrics:")
            if 'neural_results' in results and 'baseline_results' in results:
                neural_accs = [r.get('accuracy', 0) for r in results['neural_results'].values()]
                baseline_accs = [r.get('accuracy', 0) for r in results['baseline_results'].values() 
                               if isinstance(r, dict)]
                
                if neural_accs and baseline_accs:
                    best_neural = max(neural_accs)
                    best_baseline = max(baseline_accs)
                    improvement = ((best_neural - best_baseline) / best_baseline) * 100
                    
                    print(f"  Best Traditional Model Accuracy: {best_baseline:.4f}")
                    print(f"  Best Deep Learning Model Accuracy: {best_neural:.4f}")
                    print(f"  Relative Performance Improvement: {improvement:.2f}%")
            
            if 'correction_results' in results:
                correction = results['correction_results']
                error_reduction = ((correction['corrected_accuracy'] - correction['simulated_accuracy']) / 
                                 (1 - correction['simulated_accuracy'])) * 100
                print(f"  Error Rate Reduction through Correction: {error_reduction:.2f}%")
            
            print("=" * 70)
            
            return 0
        else:
            print(f"Evaluation failed: {results['error']}")
            return 1
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required modules are available")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())