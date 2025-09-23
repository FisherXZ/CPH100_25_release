#!/usr/bin/env python3
"""
Feature Importance Analysis - Visual Comparison of Top Features vs Less Important Ones
Creates comprehensive plots showing the performance impact of individual features
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Set style for publication-quality plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_ablation_results():
    """Load ablation study results from the expanded report"""
    
    # Results extracted from the comprehensive ablation study
    results = {
        'age_only': {
            'name': 'Age Only (Baseline)',
            'features': ['age'],
            'feature_count': 1,
            'val_auc': 0.6067,
            'improvement': 0.0,
            'category': 'baseline'
        },
        'age_sex': {
            'name': 'Age + Sex',
            'features': ['age', 'sex'],
            'feature_count': 3,
            'val_auc': 0.6175,
            'improvement': 0.0108,
            'category': 'demographics'
        },
        'age_demographics': {
            'name': 'Age + Demographics',
            'features': ['age', 'sex', 'race7', 'educat'],
            'feature_count': 10,
            'val_auc': 0.6304,
            'improvement': 0.0237,
            'category': 'demographics'
        },
        'age_pack_years': {
            'name': 'Age + Pack-Years',
            'features': ['age', 'pack_years'],
            'feature_count': 2,
            'val_auc': 0.8234,
            'improvement': 0.2167,
            'category': 'smoking'
        },
        'age_smoking_basic': {
            'name': 'Age + Basic Smoking',
            'features': ['age', 'pack_years', 'smoked_f', 'cig_stat'],
            'feature_count': 5,
            'val_auc': 0.8223,
            'improvement': 0.2156,
            'category': 'smoking'
        },
        'age_smoking_full': {
            'name': 'Age + Full Smoking',
            'features': ['age', 'pack_years', 'cig_years', 'cigpd_f', 'smoked_f', 'cig_stat'],
            'feature_count': 6,
            'val_auc': 0.8391,
            'improvement': 0.2324,
            'category': 'smoking'
        },
        'age_family_history': {
            'name': 'Age + Family History',
            'features': ['age', 'fh_cancer', 'lung_fh'],
            'feature_count': 4,
            'val_auc': 0.6269,
            'improvement': 0.0202,
            'category': 'medical'
        },
        'age_comorbidities': {
            'name': 'Age + Comorbidities',
            'features': ['age', 'diabetes_f', 'hearta_f', 'stroke_f'],
            'feature_count': 4,
            'val_auc': 0.6003,
            'improvement': -0.0064,
            'category': 'medical'
        },
        'full_model': {
            'name': 'Full Model',
            'features': ['all_15_variables'],
            'feature_count': 30,
            'val_auc': 0.8378,
            'improvement': 0.2311,
            'category': 'comprehensive'
        }
    }
    
    return results

def create_individual_feature_impact_analysis():
    """Analyze the impact of adding individual key features"""
    
    # Feature impact analysis based on ablation results
    feature_impacts = {
        # Top 3 most important features
        'pack_years': {
            'name': 'Pack-Years',
            'without': 0.6067,  # age only
            'with': 0.8234,     # age + pack_years
            'improvement': 0.2167,
            'category': 'Top 3',
            'rank': 1,
            'description': 'Smoking intensity over lifetime'
        },
        'age': {
            'name': 'Age',
            'without': 0.5000,  # theoretical no-information baseline
            'with': 0.6067,     # age only
            'improvement': 0.1067,
            'category': 'Top 3',
            'rank': 2,
            'description': 'Patient age (baseline demographic)'
        },
        'cig_years': {
            'name': 'Cigarette Years',
            'without': 0.8234,  # age + pack_years
            'with': 0.8391,     # age + full smoking (includes cig_years)
            'improvement': 0.0157,
            'category': 'Top 3',
            'rank': 3,
            'description': 'Years of smoking (duration component)'
        },
        
        # Less important features for comparison
        'sex': {
            'name': 'Sex',
            'without': 0.6067,  # age only
            'with': 0.6175,     # age + sex
            'improvement': 0.0108,
            'category': 'Less Important',
            'rank': 4,
            'description': 'Patient gender'
        },
        'family_history': {
            'name': 'Family History',
            'without': 0.6067,  # age only
            'with': 0.6269,     # age + family history
            'improvement': 0.0202,
            'category': 'Less Important',
            'rank': 5,
            'description': 'Cancer family history'
        },
        'demographics': {
            'name': 'Demographics',
            'without': 0.6067,  # age only
            'with': 0.6304,     # age + demographics
            'improvement': 0.0237,
            'category': 'Less Important',
            'rank': 6,
            'description': 'Race, education, etc.'
        },
        'comorbidities': {
            'name': 'Comorbidities',
            'without': 0.6067,  # age only
            'with': 0.6003,     # age + comorbidities
            'improvement': -0.0064,
            'category': 'Harmful',
            'rank': 7,
            'description': 'Diabetes, heart disease, stroke'
        }
    }
    
    return feature_impacts

def plot_feature_importance_comparison():
    """Create comprehensive feature importance comparison plots"""
    
    feature_impacts = create_individual_feature_impact_analysis()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Individual Feature Impact Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    
    features = list(feature_impacts.keys())
    improvements = [feature_impacts[f]['improvement'] for f in features]
    colors = ['#e74c3c' if imp < 0 else '#2ecc71' if imp > 0.1 else '#f39c12' if imp > 0.02 else '#95a5a6' 
              for imp in improvements]
    
    bars = ax1.bar(range(len(features)), improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUC Improvement', fontsize=12, fontweight='bold')
    ax1.set_title('Individual Feature Impact on Model Performance\n(Improvement over baseline)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(features)))
    ax1.set_xticklabels([feature_impacts[f]['name'] for f in features], 
                       rotation=45, ha='right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height >= 0 else -0.01),
                f'{imp:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=9)
    
    # 2. Before/After Performance Comparison
    ax2 = plt.subplot(2, 3, 2)
    
    # Focus on top 4 features for clarity
    top_features = ['pack_years', 'age', 'cig_years', 'sex']
    
    x_pos = np.arange(len(top_features))
    width = 0.35
    
    without_scores = [feature_impacts[f]['without'] for f in top_features]
    with_scores = [feature_impacts[f]['with'] for f in top_features]
    
    bars1 = ax2.bar(x_pos - width/2, without_scores, width, label='Without Feature', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x_pos + width/2, with_scores, width, label='With Feature', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation AUC', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Comparison: With vs Without Key Features', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([feature_impacts[f]['name'] for f in top_features], fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0.45, 0.85)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Feature Importance Ranking
    ax3 = plt.subplot(2, 3, 3)
    
    # Sort features by improvement
    sorted_features = sorted(feature_impacts.items(), key=lambda x: x[1]['improvement'], reverse=True)
    
    feature_names = [item[1]['name'] for item in sorted_features]
    improvements_sorted = [item[1]['improvement'] for item in sorted_features]
    
    # Color by category
    category_colors = {
        'Top 3': '#e74c3c',
        'Less Important': '#f39c12', 
        'Harmful': '#95a5a6'
    }
    colors_sorted = [category_colors.get(item[1]['category'], '#3498db') for item in sorted_features]
    
    bars = ax3.barh(range(len(feature_names)), improvements_sorted, color=colors_sorted, 
                   alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_yticks(range(len(feature_names)))
    ax3.set_yticklabels(feature_names, fontsize=10)
    ax3.set_xlabel('AUC Improvement', fontsize=12, fontweight='bold')
    ax3.set_title('Feature Importance Ranking\n(Sorted by Performance Impact)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax3.grid(axis='x', alpha=0.3)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements_sorted)):
        width = bar.get_width()
        ax3.text(width + (0.002 if width >= 0 else -0.002), bar.get_y() + bar.get_height()/2.,
                f'{imp:.3f}', ha='left' if width >= 0 else 'right', va='center', 
                fontweight='bold', fontsize=9)
    
    # 4. Cumulative Feature Impact
    ax4 = plt.subplot(2, 3, 4)
    
    # Show cumulative impact of adding features
    cumulative_features = ['Age Only', 'Age + Sex', 'Age + Demographics', 'Age + Pack-Years', 
                          'Age + Full Smoking', 'Full Model']
    cumulative_aucs = [0.6067, 0.6175, 0.6304, 0.8234, 0.8391, 0.8378]
    
    ax4.plot(cumulative_features, cumulative_aucs, marker='o', linewidth=3, markersize=8,
            color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
    ax4.set_ylabel('Validation AUC', fontsize=12, fontweight='bold')
    ax4.set_title('Cumulative Impact of Adding Features\n(Sequential Feature Addition)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax4.set_xticklabels(cumulative_features, rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.55, 0.85)
    
    # Highlight the dramatic jump at pack-years
    ax4.annotate('Massive Jump\n+21.7% AUC!', 
                xy=(3, 0.8234), xytext=(2, 0.75),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red',
                ha='center')
    
    # Add value labels
    for i, auc in enumerate(cumulative_aucs):
        ax4.text(i, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # 5. Feature Category Performance Summary
    ax5 = plt.subplot(2, 3, 5)
    
    categories = ['Baseline\n(Age Only)', 'Demographics\n(+Sex, Race, Edu)', 
                 'Smoking\n(+Pack-Years)', 'Medical\n(+Family Hist)', 'Full Model\n(All Features)']
    category_aucs = [0.6067, 0.6304, 0.8391, 0.6269, 0.8378]
    category_colors_list = ['#95a5a6', '#f39c12', '#e74c3c', '#3498db', '#9b59b6']
    
    bars = ax5.bar(categories, category_aucs, color=category_colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    ax5.set_ylabel('Validation AUC', fontsize=12, fontweight='bold')
    ax5.set_title('Performance by Feature Category\n(Best Model per Category)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax5.set_xticklabels(categories, fontsize=9, ha='center')
    ax5.grid(axis='y', alpha=0.3)
    ax5.set_ylim(0.55, 0.85)
    
    # Add value labels and highlight best
    for i, (bar, auc) in enumerate(zip(bars, category_aucs)):
        ax5.text(bar.get_x() + bar.get_width()/2., auc + 0.005,
                f'{auc:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10,
                color='red' if auc > 0.83 else 'black')
    
    # 6. Feature Effectiveness Scatter Plot
    ax6 = plt.subplot(2, 3, 6)
    
    # Plot feature count vs improvement
    feature_counts = [1, 3, 10, 2, 6, 4, 4, 30]
    improvements_list = [0, 0.0108, 0.0237, 0.2167, 0.2324, 0.0202, -0.0064, 0.2311]
    feature_labels = ['Age Only', 'Age+Sex', 'Age+Demo', 'Age+Pack-Yrs', 
                     'Age+Smoking', 'Age+Family', 'Age+Comorbid', 'Full Model']
    
    # Color by performance level
    colors_scatter = ['#e74c3c' if imp > 0.2 else '#f39c12' if imp > 0.02 else '#95a5a6' 
                     for imp in improvements_list]
    
    scatter = ax6.scatter(feature_counts, improvements_list, c=colors_scatter, s=100, 
                         alpha=0.8, edgecolors='black', linewidth=1)
    
    ax6.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax6.set_ylabel('AUC Improvement', fontsize=12, fontweight='bold')
    ax6.set_title('Feature Efficiency Analysis\n(Performance vs Complexity)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax6.grid(True, alpha=0.3)
    
    # Add labels for key points
    for i, label in enumerate(feature_labels):
        if improvements_list[i] > 0.2 or feature_counts[i] in [1, 2, 30]:
            ax6.annotate(label, (feature_counts[i], improvements_list[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, ha='left')
    
    # Highlight the efficiency sweet spot
    ax6.annotate('Efficiency\nSweet Spot!', 
                xy=(2, 0.2167), xytext=(15, 0.15),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red',
                ha='center')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('feature_importance_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_top3_detailed_comparison():
    """Create detailed comparison of top 3 features vs others"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Data for top 3 vs others comparison
    top3_features = ['Pack-Years', 'Age', 'Cigarette Years']
    top3_improvements = [0.2167, 0.1067, 0.0157]
    
    other_features = ['Sex', 'Family History', 'Demographics', 'Comorbidities']
    other_improvements = [0.0108, 0.0202, 0.0237, -0.0064]
    
    # 1. Direct comparison bar chart
    all_features = top3_features + other_features
    all_improvements = top3_improvements + other_improvements
    colors = ['#e74c3c'] * 3 + ['#95a5a6'] * 4
    
    bars = ax1.bar(all_features, all_improvements, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('AUC Improvement', fontsize=12, fontweight='bold')
    ax1.set_title('Top 3 Features vs Others\n(Individual Performance Impact)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticklabels(all_features, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels
    for bar, imp in zip(bars, all_improvements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height >= 0 else -0.01),
                f'{imp:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=9)
    
    # Add category separation line
    ax1.axvline(x=2.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(1, 0.18, 'TOP 3', ha='center', fontsize=12, fontweight='bold', color='red')
    ax1.text(5, 0.18, 'OTHERS', ha='center', fontsize=12, fontweight='bold', color='gray')
    
    # 2. Cumulative importance
    cumulative_names = ['Age', 'Age + Pack-Years', 'Age + Pack-Years + Cig-Years', 'Add Other Features']
    cumulative_values = [0.6067, 0.8234, 0.8391, 0.8378]
    
    ax2.plot(cumulative_names, cumulative_values, marker='o', linewidth=3, markersize=10,
            color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
    ax2.set_ylabel('Validation AUC', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Impact of Top 3 Features\n(Sequential Addition)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticklabels(cumulative_names, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.55, 0.85)
    
    # Highlight the pack-years jump
    ax2.annotate('Pack-Years\nAdds 21.7%!', 
                xy=(1, 0.8234), xytext=(0.5, 0.75),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red')
    
    # 3. Performance distribution
    top3_data = [0.2167, 0.1067, 0.0157]
    others_data = [0.0108, 0.0202, 0.0237, -0.0064]
    
    ax3.boxplot([top3_data, others_data], labels=['Top 3 Features', 'Other Features'],
               patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax3.set_ylabel('AUC Improvement', fontsize=12, fontweight='bold')
    ax3.set_title('Performance Distribution Comparison\n(Box Plot Analysis)', 
                  fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add individual points
    ax3.scatter([1]*len(top3_data), top3_data, color='red', s=50, alpha=0.8, zorder=3)
    ax3.scatter([2]*len(others_data), others_data, color='gray', s=50, alpha=0.8, zorder=3)
    
    # 4. Relative importance pie chart
    total_improvement = sum([abs(x) for x in top3_improvements + other_improvements if x > 0])
    top3_contribution = sum(top3_improvements) / total_improvement * 100
    others_contribution = sum([x for x in other_improvements if x > 0]) / total_improvement * 100
    
    sizes = [top3_contribution, others_contribution]
    labels = [f'Top 3 Features\n({top3_contribution:.1f}%)', f'Other Features\n({others_contribution:.1f}%)']
    colors_pie = ['#e74c3c', '#95a5a6']
    
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax4.set_title('Relative Contribution to Model Performance\n(Total Positive Impact)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('top3_features_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main function to create all feature importance visualizations"""
    
    print("Creating comprehensive feature importance analysis...")
    
    # Create comprehensive analysis
    fig1 = plot_feature_importance_comparison()
    print("✅ Created comprehensive feature importance analysis")
    
    # Create detailed top 3 comparison
    fig2 = create_top3_detailed_comparison()
    print("✅ Created detailed top 3 features comparison")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS SUMMARY")
    print("="*60)
    
    feature_impacts = create_individual_feature_impact_analysis()
    
    print("\nTOP 3 MOST IMPORTANT FEATURES:")
    top3 = sorted([(k, v) for k, v in feature_impacts.items() if v['category'] == 'Top 3'], 
                  key=lambda x: x[1]['improvement'], reverse=True)
    
    for i, (feature, data) in enumerate(top3, 1):
        print(f"{i}. {data['name']}: +{data['improvement']:.3f} AUC improvement")
        print(f"   Description: {data['description']}")
        print(f"   Performance: {data['without']:.3f} → {data['with']:.3f}")
        print()
    
    print("COMPARISON WITH LESS IMPORTANT FEATURES:")
    others = [v for v in feature_impacts.values() if v['category'] in ['Less Important', 'Harmful']]
    others.sort(key=lambda x: x['improvement'], reverse=True)
    
    for data in others:
        print(f"• {data['name']}: {data['improvement']:+.3f} AUC improvement")
    
    print(f"\nKEY INSIGHT:")
    print(f"Top 3 features contribute {sum(d['improvement'] for d in top3):.3f} AUC improvement")
    print(f"All other features contribute {sum(d['improvement'] for d in others if d['improvement'] > 0):.3f} AUC improvement")
    print(f"Top 3 features are {(sum(d['improvement'] for d in top3) / sum(d['improvement'] for d in others if d['improvement'] > 0)):.1f}x more impactful!")

if __name__ == "__main__":
    main()
