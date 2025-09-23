"""
Fixed Subgroup Analysis - Including Race
Corrected version that properly handles all subgroups including race
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from csv import DictReader
import random

from vectorizer import Vectorizer
from logistic_regression import LogisticRegression

def run_complete_subgroup_analysis():
    """Run corrected subgroup analysis including race."""
    print("🎯 Fixed Subgroup Analysis - Including Race")
    print("="*60)
    
    # Setup data and model (same as before)
    print("Setting up model and data...")
    reader = DictReader(open("lung_prsn.csv", "r"))
    rows = [r for r in reader]
    
    NUM_TRAIN, NUM_VAL = 100000, 25000
    random.seed(0)
    random.shuffle(rows)
    
    train_data = rows[:NUM_TRAIN]
    val_data = rows[NUM_TRAIN:NUM_TRAIN+NUM_VAL] 
    test_data = rows[NUM_TRAIN+NUM_VAL:]
    
    # Train model
    feature_config = {
        "numerical": ["age", "pack_years", "cig_years", "cigpd_f"],
        "categorical": ["smoked_f"],
        "ordinal": ["cig_stat"]
    }
    
    model = LogisticRegression(
        num_epochs=200, learning_rate=0.001, 
        batch_size=128, regularization_lambda=0, verbose=False
    )
    
    vectorizer = Vectorizer(feature_config)
    vectorizer.fit(train_data)
    
    train_X = vectorizer.transform(train_data)
    val_X = vectorizer.transform(val_data)
    train_Y = np.array([int(r["lung_cancer"]) for r in train_data])
    val_Y = np.array([int(r["lung_cancer"]) for r in val_data])
    
    model.fit(train_X, train_Y, val_X, val_Y)
    
    # Generate test predictions
    test_X = vectorizer.transform(test_data)
    test_labels = np.array([int(r["lung_cancer"]) for r in test_data])
    test_predictions = model.predict_proba(test_X)
    
    overall_auc = roc_auc_score(test_labels, test_predictions)
    print(f"Overall Test AUC: {overall_auc:.4f}")
    
    # Define subgroups with proper race implementation
    subgroups = {
        'Sex': {
            'Male': lambda r: r.get('sex', '') == '1',
            'Female': lambda r: r.get('sex', '') == '2'
        },
        'Race': {
            'White': lambda r: r.get('race7', '') == '1',
            'Black': lambda r: r.get('race7', '') == '2',
            'Hispanic': lambda r: r.get('race7', '') == '3',
            'Asian': lambda r: r.get('race7', '') == '4',
            'Other': lambda r: r.get('race7', '') in ['5', '6', '7']
        },
        'Education': {
            'High School or Less': lambda r: r.get('educat', '') in ['1', '2', '3'],
            'Some College': lambda r: r.get('educat', '') in ['4', '5'],
            'College Graduate+': lambda r: r.get('educat', '') in ['6', '7']
        },
        'Smoking Status': {
            'Never': lambda r: r.get('cig_stat', '') == '0',
            'Former': lambda r: r.get('cig_stat', '') == '1', 
            'Current': lambda r: r.get('cig_stat', '') == '2'
        },
        'NLST Eligibility': {
            'NLST Eligible': lambda r: r.get('nlst_flag', '') == '1',
            'Not Eligible': lambda r: r.get('nlst_flag', '') in ['0', '']
        }
    }
    
    # Analyze all subgroups
    print("\nAnalyzing all subgroups...")
    results = {}
    
    for category, groups in subgroups.items():
        print(f"\n{category}:")
        category_results = {}
        
        for group_name, condition in groups.items():
            mask = np.array([condition(r) for r in test_data])
            
            if np.sum(mask) < 50:  # Lower threshold for race analysis
                print(f"  {group_name}: Skipped (N={np.sum(mask)})")
                continue
            
            group_labels = test_labels[mask]
            group_predictions = test_predictions[mask]
            
            if len(np.unique(group_labels)) < 2:
                print(f"  {group_name}: Skipped (no outcome variation)")
                continue
            
            group_auc = roc_auc_score(group_labels, group_predictions)
            cancer_rate = np.mean(group_labels)
            
            category_results[group_name] = {
                'auc': group_auc,
                'n_patients': np.sum(mask),
                'n_cancer': np.sum(group_labels),
                'cancer_rate': cancer_rate,
                'mask': mask  # Store mask for plotting
            }
            
            print(f"  {group_name}: AUC={group_auc:.3f}, N={np.sum(mask):,}, "
                  f"Cancer={np.sum(group_labels)} ({cancer_rate:.1%})")
        
        results[category] = category_results
    
    # Create comprehensive plot
    print("\nGenerating comprehensive subgroup ROC plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for idx, (category, groups) in enumerate(results.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        for i, (group_name, stats) in enumerate(groups.items()):
            # Use stored mask to get ROC curve
            mask = stats['mask']
            group_labels = test_labels[mask]
            group_predictions = test_predictions[mask]
            
            fpr, tpr, _ = roc_curve(group_labels, group_predictions)
            
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5,
                   label=f'{group_name} (AUC={stats["auc"]:.3f})')
        
        # Random classifier
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(category, fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Complete Subgroup Analysis: Model Performance by Demographics', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("complete_subgroup_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Analyze disparities
    print("\n" + "="*60)
    print("COMPLETE SUBGROUP DISPARITY ANALYSIS")
    print("="*60)
    
    for category, groups in results.items():
        if len(groups) < 2:
            continue
            
        aucs = [stats['auc'] for stats in groups.values()]
        group_names = list(groups.keys())
        
        max_auc = max(aucs)
        min_auc = min(aucs)
        auc_range = max_auc - min_auc
        
        max_group = group_names[aucs.index(max_auc)]
        min_group = group_names[aucs.index(min_auc)]
        
        print(f"\n{category.upper()}:")
        print(f"  Best: {max_group} (AUC = {max_auc:.3f})")
        print(f"  Worst: {min_group} (AUC = {min_auc:.3f})")
        print(f"  Difference: {auc_range:.3f}")
        
        if auc_range > 0.05:
            print(f"  ⚠️  CONCERNING: >0.05 AUC difference")
        else:
            print(f"  ✅ Acceptable: <0.05 AUC difference")
    
    print(f"\n🎉 Complete Analysis Finished!")
    print(f"Generated: complete_subgroup_analysis.png")

if __name__ == "__main__":
    run_complete_subgroup_analysis()
