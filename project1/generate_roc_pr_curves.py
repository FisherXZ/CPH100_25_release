"""
Generate ROC and Precision-Recall Curves for Best Model
Task 2.2: Model Performance Analysis with Clinical Curves
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from csv import DictReader
import random

from vectorizer import Vectorizer
from logistic_regression import LogisticRegression

def load_and_split_data(data_path="lung_prsn.csv"):
    """Load PLCO data with consistent splits."""
    print("Loading PLCO data...")
    reader = DictReader(open(data_path, "r"))
    rows = [r for r in reader]
    
    # Same split as main.py and ablation study
    NUM_TRAIN, NUM_VAL = 100000, 25000
    random.seed(0)  # Consistent seed
    random.shuffle(rows)
    
    train_data = rows[:NUM_TRAIN]
    val_data = rows[NUM_TRAIN:NUM_TRAIN+NUM_VAL] 
    test_data = rows[NUM_TRAIN+NUM_VAL:]
    
    print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    return train_data, val_data, test_data

def train_best_model(train_data, val_data):
    """Train our best model (age + comprehensive smoking history)."""
    print("Training best model (6 smoking features)...")
    
    # Best feature configuration from ablation study
    feature_config = {
        "numerical": ["age", "pack_years", "cig_years", "cigpd_f"],
        "categorical": ["smoked_f"],
        "ordinal": ["cig_stat"]
    }
    
    # Optimal hyperparameters from grid search
    model = LogisticRegression(
        num_epochs=200,
        learning_rate=0.001,
        batch_size=128,
        regularization_lambda=0,
        verbose=True
    )
    
    # Prepare data
    vectorizer = Vectorizer(feature_config)
    vectorizer.fit(train_data)
    
    train_X = vectorizer.transform(train_data)
    val_X = vectorizer.transform(val_data)
    
    train_Y = np.array([int(r["lung_cancer"]) for r in train_data])
    val_Y = np.array([int(r["lung_cancer"]) for r in val_data])
    
    # Train model
    model.fit(train_X, train_Y, val_X, val_Y)
    
    # Evaluate on validation set
    val_pred = model.predict_proba(val_X)
    val_auc = roc_auc_score(val_Y, val_pred)
    print(f"Validation AUC: {val_auc:.4f}")
    
    return model, vectorizer

def generate_curves(model, vectorizer, test_data):
    """Generate ROC and PR curves for the test set."""
    print("Generating ROC and PR curves on test set...")
    
    # Prepare test data
    test_X = vectorizer.transform(test_data)
    test_Y = np.array([int(r["lung_cancer"]) for r in test_data])
    
    # Get predictions
    pred_test_Y = model.predict_proba(test_X)
    
    # Calculate curves
    fpr, tpr, roc_thresholds = roc_curve(test_Y, pred_test_Y)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(test_Y, pred_test_Y)
    pr_auc = auc(recall, precision)
    
    # Create figure with both curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.6, label='Random classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve - Test Set Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add performance text
    ax1.text(0.6, 0.2, f'Test AUC: {roc_auc:.1%}\n6 Features\n(Age + Smoking)', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             fontsize=11, fontweight='bold')
    
    # Precision-Recall Curve  
    ax2.plot(recall, precision, color='blue', lw=3, label=f'PR curve (AUC = {pr_auc:.4f})')
    
    # Add baseline (random classifier for imbalanced data)
    cancer_rate = np.mean(test_Y)
    ax2.axhline(y=cancer_rate, color='navy', lw=2, linestyle='--', alpha=0.6, 
                label=f'Random classifier (baseline = {cancer_rate:.3f})')
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve - Test Set Performance', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    
    # Add performance text
    ax2.text(0.05, 0.95, f'PR AUC: {pr_auc:.3f}\nCancer Rate: {cancer_rate:.1%}\nTest Cases: {len(test_Y):,}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
             fontsize=11, fontweight='bold', verticalalignment='top')
    
    # Overall title
    fig.suptitle('CPH 100A Project 1: Best Model Performance Analysis\nAge + Comprehensive Smoking History (6 Features)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save individual curves
    plt.savefig("roc_pr_curves_combined.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Save individual ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Best Model Test Performance', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Save individual PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=3, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.axhline(y=cancer_rate, color='navy', lw=2, linestyle='--', alpha=0.6, 
                label=f'Random baseline ({cancer_rate:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Best Model Test Performance', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.savefig("pr_curve.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Print detailed results
    print("\n" + "="*60)
    print("TEST SET PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Model: Age + Comprehensive Smoking History (6 features)")
    print(f"Test Set Size: {len(test_Y):,} patients")
    print(f"Cancer Cases: {np.sum(test_Y):,} ({cancer_rate:.1%})")
    print(f"Non-cancer Cases: {len(test_Y) - np.sum(test_Y):,} ({1-cancer_rate:.1%})")
    print(f"\nPerformance Metrics:")
    print(f"  ROC AUC: {roc_auc:.4f} ({roc_auc:.1%})")
    print(f"  PR AUC:  {pr_auc:.4f}")
    print(f"  Baseline (random): {cancer_rate:.4f}")
    print(f"\nClinical Interpretation:")
    if roc_auc >= 0.8:
        print(f"  • Excellent discrimination (AUC ≥ 0.8)")
    elif roc_auc >= 0.7:
        print(f"  • Good discrimination (0.7 ≤ AUC < 0.8)")
    else:
        print(f"  • Fair discrimination (AUC < 0.7)")
    
    improvement_factor = pr_auc / cancer_rate
    print(f"  • PR AUC is {improvement_factor:.1f}x better than random")
    print("="*60)
    
    return roc_auc, pr_auc, cancer_rate

def main():
    """Main function to generate ROC and PR curves."""
    print("🎯 Generating ROC and PR Curves for Best Model")
    print("="*55)
    
    # Load data
    train_data, val_data, test_data = load_and_split_data()
    
    # Train best model
    model, vectorizer = train_best_model(train_data, val_data)
    
    # Generate curves
    roc_auc, pr_auc, cancer_rate = generate_curves(model, vectorizer, test_data)
    
    print(f"\n🎉 Analysis Complete!")
    print(f"Generated files:")
    print(f"  • roc_pr_curves_combined.png (both curves)")
    print(f"  • roc_curve.png (ROC only)")  
    print(f"  • pr_curve.png (PR only)")
    print("="*55)

if __name__ == "__main__":
    main()
