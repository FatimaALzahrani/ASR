class ResultsComparator:
    def __init__(self):
        self.previous_results = {
            'Basic Model': 45.42,
            'Without Aseel': 46.77,
            'Without Ahmed': 42.53,
            'Asem Specialized Model': 55.56,
            'Aseel Specialized Model': 50.56,
            'Wasam Specialized Model': 48.89,
            'Haifa Specialized Model': 47.62,
            'Ahmed Specialized Model': 40.77
        }
    
    def compare_results(self, enhanced_results):
        print(f"Final Results Comparison")
        print("="*50)
        
        comparison_results = {}
        
        comparisons = [
            ('Basic Model', 'Enhanced Basic Model'),
            ('Without Aseel', 'Enhanced Without Aseel'),
            ('Without Ahmed', 'Enhanced Without Ahmed'),
            ('Asem Specialized Model', 'Enhanced Asem Specialized Model'),
            ('Aseel Specialized Model', 'Enhanced Aseel Specialized Model')
        ]
        
        for old_model, new_model in comparisons:
            if old_model in self.previous_results and new_model in enhanced_results:
                old_acc = self.previous_results[old_model]
                new_acc = enhanced_results[new_model]
                improvement = new_acc - old_acc
                improvement_pct = (improvement / old_acc) * 100 if old_acc > 0 else 0
                
                comparison_results[old_model] = {
                    'old_accuracy': old_acc,
                    'new_accuracy': new_acc,
                    'improvement': improvement,
                    'improvement_percentage': improvement_pct
                }
                
                if improvement > 0:
                    emoji = "📈"
                elif improvement < 0:
                    emoji = "📉"
                else:
                    emoji = "➡️"
                
                print(f"{emoji} {old_model}:")
                print(f"   Before enhancement: {old_acc:.2f}%")
                print(f"   After enhancement: {new_acc:.2f}%")
                print(f"   Improvement: {improvement:+.2f}% ({improvement_pct:+.1f}%)")
                print()
        
        return comparison_results