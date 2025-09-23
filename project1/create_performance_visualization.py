"""
Create Performance Visualization for Ablation Study
Shows model performance across different feature combinations
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

def create_performance_visualization():
    """Create comprehensive visualization of ablation study results."""
    
    # Data from our ablation experiments
    experiments = [
        ("Age Only", 1, 60.67, "Baseline"),
        ("Age + Sex", 3, 61.75, "Demographics"),
        ("Age + Demographics", 10, 63.04, "Demographics"), 
        ("Age + Pack-Years", 2, 82.34, "Smoking"),
        ("Age + Smoking Basic", 5, 82.23, "Smoking"),
        ("Age + Smoking Full", 6, 83.91, "Smoking"),
        ("Age + Family History", 4, 62.69, "Medical History"),
        ("Age + Comorbidities", 4, 60.03, "Medical History"),
        ("Demographics + Smoking", 15, 83.45, "Combined"),
        ("Full Model", 30, 83.78, "Combined")
    ]
    
    # Extract data for plotting
    names = [exp[0] for exp in experiments]
    feature_counts = [exp[1] for exp in experiments]
    aucs = [exp[2] for exp in experiments]
    categories = [exp[3] for exp in experiments]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Define colors for categories
    color_map = {
        "Baseline": "#FF6B6B",
        "Demographics": "#4ECDC4", 
        "Smoking": "#45B7D1",
        "Medical History": "#96CEB4",
        "Combined": "#FFEAA7"
    }
    colors = [color_map[cat] for cat in categories]
    
    # Plot 1: AUC Performance by Experiment
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(names)), aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.title('Model Performance Across Feature Combinations', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Combination', fontsize=12)
    plt.ylabel('Validation AUC (%)', fontsize=12)
    plt.xticks(range(len(names)), [name.replace(' + ', '\n+\n') for name in names], 
               rotation=45, ha='right', fontsize=10)
    plt.ylim(55, 85)
    
    # Add value labels on bars
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{auc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Highlight best model
    best_idx = aucs.index(max(aucs))
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Plot 2: Feature Count vs Performance
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(feature_counts, aucs, c=colors, s=150, alpha=0.8, 
                         edgecolors='black', linewidth=1)
    plt.title('Feature Count vs Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Validation AUC (%)', fontsize=12)
    
    # Add labels for key points
    for i, (name, count, auc) in enumerate(zip(names, feature_counts, aucs)):
        if name in ["Age Only", "Age + Pack-Years", "Age + Smoking Full", "Full Model"]:
            plt.annotate(name.replace(" + ", "\n"), (count, auc), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, ha='left')
    
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Performance Improvement Over Baseline
    plt.subplot(2, 2, 3)
    baseline_auc = 60.67
    improvements = [auc - baseline_auc for auc in aucs]
    
    bars = plt.bar(range(len(names)), improvements, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    plt.title('AUC Improvement Over Age-Only Baseline', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Combination', fontsize=12)
    plt.ylabel('AUC Improvement (percentage points)', fontsize=12)
    plt.xticks(range(len(names)), [name.replace(' + ', '\n+\n') for name in names], 
               rotation=45, ha='right', fontsize=10)
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'+{imp:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Highlight the big jump from pack-years
    pack_years_idx = names.index("Age + Pack-Years")
    bars[pack_years_idx].set_edgecolor('red')
    bars[pack_years_idx].set_linewidth(3)
    
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 4: Category Performance Summary
    plt.subplot(2, 2, 4)
    
    # Group by category and get best performance in each
    category_performance = {}
    for name, count, auc, cat in experiments:
        if cat not in category_performance:
            category_performance[cat] = []
        category_performance[cat].append(auc)
    
    # Get max AUC for each category
    cat_names = list(category_performance.keys())
    cat_max_aucs = [max(category_performance[cat]) for cat in cat_names]
    cat_colors = [color_map[cat] for cat in cat_names]
    
    bars = plt.bar(cat_names, cat_max_aucs, color=cat_colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    plt.title('Best Performance by Feature Category', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Category', fontsize=12)
    plt.ylabel('Best Validation AUC (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, auc in zip(bars, cat_max_aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{auc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(55, 85)
    
    # Add overall title and legend
    fig.suptitle('CPH 100A Project 1: Ablation Study Results\nModel Performance Across Feature Combinations', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[cat], alpha=0.8, 
                                   edgecolor='black') for cat in color_map.keys()]
    fig.legend(legend_elements, color_map.keys(), loc='center', 
               bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)
    
    # Save the plot
    plt.savefig('ablation_study_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ABLATION STUDY PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"\n🏆 BEST PERFORMING MODEL:")
    best_idx = aucs.index(max(aucs))
    print(f"   {names[best_idx]}: {aucs[best_idx]:.2f}% AUC ({feature_counts[best_idx]} features)")
    
    print(f"\n📊 KEY INSIGHTS:")
    print(f"   • Baseline (Age only): {aucs[0]:.2f}% AUC")
    pack_years_idx = names.index("Age + Pack-Years")
    print(f"   • Adding pack-years: {aucs[pack_years_idx]:.2f}% AUC (+{aucs[pack_years_idx]-aucs[0]:.1f} points)")
    print(f"   • Best smoking model: {max(aucs):.2f}% AUC (+{max(aucs)-aucs[0]:.1f} points)")
    
    full_model_idx = names.index("Full Model")
    print(f"   • Full model (30 features): {aucs[full_model_idx]:.2f}% AUC")
    print(f"   • Best model (6 features): {max(aucs):.2f}% AUC")
    print(f"   • Quality > Quantity: {max(aucs) - aucs[full_model_idx]:.2f} point advantage")
    
    print(f"\n🚬 SMOKING DOMINANCE:")
    smoking_improvement = aucs[pack_years_idx] - aucs[0]
    total_improvement = max(aucs) - aucs[0]
    smoking_contribution = (smoking_improvement / total_improvement) * 100
    print(f"   • Smoking features provide {smoking_contribution:.0f}% of total improvement")
    
    print("="*60)

if __name__ == "__main__":
    create_performance_visualization()
