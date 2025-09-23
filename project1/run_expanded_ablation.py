"""
Run Expanded Feature Ablation Analysis
CPH 100A Project 1 - Task 2.1

This script runs the expanded feature comparison and generates visualizations.
"""

from ablation_analysis import AblationAnalyzer

def main():
    """
    Run the expanded feature ablation analysis.
    """
    print("🎯 Starting Expanded Feature Ablation Analysis")
    print("=" * 55)
    print("This will test 10 different feature combinations:")
    print("1. age_only")
    print("2. age_sex") 
    print("3. age_demographics")
    print("4. age_pack_years")
    print("5. age_smoking_basic")
    print("6. age_smoking_full")
    print("7. age_family_history")
    print("8. age_comorbidities")
    print("9. demographics_smoking")
    print("10. full_model")
    print("=" * 55)
    
    # Initialize analyzer
    analyzer = AblationAnalyzer()
    
    # Load data
    analyzer.load_and_split_data()
    
    # Run expanded feature comparison
    analyzer.compare_feature_sets_expanded()
    
    # Generate visualizations
    print("\n📊 Generating learning curves and analysis plots...")
    analyzer.plot_learning_curves("expanded_feature_analysis.png")
    
    # Generate analysis report
    print("\n📝 Generating detailed analysis report...")
    analyzer.generate_analysis_report("expanded_ablation_report.txt")
    
    print("\n" + "=" * 55)
    print("🎉 Expanded Feature Ablation Analysis Complete!")
    print("Generated files:")
    print("- expanded_feature_analysis.png")
    print("- expanded_ablation_report.txt")
    print("=" * 55)
    
    # Print quick summary
    print("\n📈 Quick Results Summary:")
    print("-" * 25)
    
    # Sort experiments by validation AUC
    results_sorted = sorted(analyzer.experiment_results.items(), 
                           key=lambda x: x[1]['val_auc'], reverse=True)
    
    for i, (exp_name, results) in enumerate(results_sorted[:5]):
        print(f"{i+1}. {exp_name}: {results['val_auc']:.4f} AUC ({results['num_features']} features)")

if __name__ == "__main__":
    main()
