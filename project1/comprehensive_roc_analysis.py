"""
Comprehensive ROC and Subgroup Analysis for CPH 100A Project 1
Task 2.2: ROC/PR curves with NLST operating point and subgroup analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from csv import DictReader
import random
import pandas as pd
from scipy import stats

from vectorizer import Vectorizer
from logistic_regression import LogisticRegression

class ComprehensiveROCAnalyzer:
    """Clean, modular analyzer for ROC curves with NLST and subgroup analysis."""
    
    def __init__(self, data_path="lung_prsn.csv"):
        self.data_path = data_path
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.model = None
        self.vectorizer = None
        self.test_predictions = None
        self.test_labels = None
        
    def load_data(self):
        """Load and split PLCO data consistently."""
        print("Loading PLCO data...")
        reader = DictReader(open(self.data_path, "r"))
        rows = [r for r in reader]
        
        NUM_TRAIN, NUM_VAL = 100000, 25000
        random.seed(0)
        random.shuffle(rows)
        
        self.train_data = rows[:NUM_TRAIN]
        self.val_data = rows[NUM_TRAIN:NUM_TRAIN+NUM_VAL] 
        self.test_data = rows[NUM_TRAIN+NUM_VAL:]
        
        print(f"Data split: {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test")
        
    def train_best_model(self):
        """Train optimal model from ablation study."""
        print("Training best model (age + comprehensive smoking)...")
        
        feature_config = {
            "numerical": ["age", "pack_years", "cig_years", "cigpd_f"],
            "categorical": ["smoked_f"],
            "ordinal": ["cig_stat"]
        }
        
        self.model = LogisticRegression(
            num_epochs=200, learning_rate=0.001, 
            batch_size=128, regularization_lambda=0, verbose=False
        )
        
        self.vectorizer = Vectorizer(feature_config)
        self.vectorizer.fit(self.train_data)
        
        train_X = self.vectorizer.transform(self.train_data)
        val_X = self.vectorizer.transform(self.val_data)
        train_Y = np.array([int(r["lung_cancer"]) for r in self.train_data])
        val_Y = np.array([int(r["lung_cancer"]) for r in self.val_data])
        
        self.model.fit(train_X, train_Y, val_X, val_Y)
        
        val_pred = self.model.predict_proba(val_X)
        val_auc = roc_auc_score(val_Y, val_pred)
        print(f"Validation AUC: {val_auc:.4f}")
        
    def generate_test_predictions(self):
        """Generate predictions on test set."""
        test_X = self.vectorizer.transform(self.test_data)
        self.test_labels = np.array([int(r["lung_cancer"]) for r in self.test_data])
        self.test_predictions = self.model.predict_proba(test_X)
        
        test_auc = roc_auc_score(self.test_labels, self.test_predictions)
        print(f"Test AUC: {test_auc:.4f}")
        
    def analyze_nlst_operating_point(self):
        """Analyze NLST criteria performance and find operating point."""
        print("\nAnalyzing NLST operating point...")
        
        # Extract NLST flags (handle empty strings)
        nlst_flags = np.array([int(r.get("nlst_flag", "0")) if r.get("nlst_flag", "0") != "" else 0 for r in self.test_data])
        
        # NLST performance: sensitivity and specificity
        nlst_tp = np.sum((nlst_flags == 1) & (self.test_labels == 1))  # True positives
        nlst_fp = np.sum((nlst_flags == 1) & (self.test_labels == 0))  # False positives
        nlst_tn = np.sum((nlst_flags == 0) & (self.test_labels == 0))  # True negatives
        nlst_fn = np.sum((nlst_flags == 0) & (self.test_labels == 1))  # False negatives
        
        nlst_sensitivity = nlst_tp / (nlst_tp + nlst_fn) if (nlst_tp + nlst_fn) > 0 else 0
        nlst_specificity = nlst_tn / (nlst_tn + nlst_fp) if (nlst_tn + nlst_fp) > 0 else 0
        nlst_fpr = 1 - nlst_specificity
        
        # NLST positive predictive value
        nlst_ppv = nlst_tp / (nlst_tp + nlst_fp) if (nlst_tp + nlst_fp) > 0 else 0
        
        nlst_stats = {
            'sensitivity': nlst_sensitivity,
            'specificity': nlst_specificity,
            'fpr': nlst_fpr,
            'ppv': nlst_ppv,
            'eligible_count': np.sum(nlst_flags),
            'eligible_rate': np.mean(nlst_flags)
        }
        
        print(f"NLST Criteria Performance:")
        print(f"  Sensitivity: {nlst_sensitivity:.3f}")
        print(f"  Specificity: {nlst_specificity:.3f}")
        print(f"  PPV: {nlst_ppv:.3f}")
        print(f"  Eligible patients: {nlst_stats['eligible_count']:,} ({nlst_stats['eligible_rate']:.1%})")
        
        return nlst_stats
        
    def plot_roc_with_nlst(self, nlst_stats):
        """Generate ROC curve with NLST operating point highlighted."""
        fpr, tpr, _ = roc_curve(self.test_labels, self.test_predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'Our Model (AUC = {roc_auc:.3f})')
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.6,
                label='Random Classifier')
        
        # Highlight NLST operating point
        plt.scatter(nlst_stats['fpr'], nlst_stats['sensitivity'], 
                   color='red', s=150, marker='*', zorder=5,
                   label=f'NLST Criteria (Sens={nlst_stats["sensitivity"]:.3f})')
        
        # Find our model's performance at NLST specificity
        target_fpr = nlst_stats['fpr']
        closest_idx = np.argmin(np.abs(fpr - target_fpr))
        our_sensitivity_at_nlst_spec = tpr[closest_idx]
        
        plt.scatter(fpr[closest_idx], our_sensitivity_at_nlst_spec,
                   color='blue', s=150, marker='o', zorder=5,
                   label=f'Our Model at NLST Spec (Sens={our_sensitivity_at_nlst_spec:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curve with NLST Operating Point Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add performance comparison text
        improvement = our_sensitivity_at_nlst_spec - nlst_stats['sensitivity']
        plt.text(0.6, 0.2, 
                f'At NLST Specificity ({nlst_stats["specificity"]:.3f}):\n'
                f'NLST Sensitivity: {nlst_stats["sensitivity"]:.3f}\n'
                f'Our Model: {our_sensitivity_at_nlst_spec:.3f}\n'
                f'Improvement: +{improvement:.3f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig("roc_curve_with_nlst.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        return {
            'our_model_auc': roc_auc,
            'our_sensitivity_at_nlst_spec': our_sensitivity_at_nlst_spec,
            'sensitivity_improvement': improvement
        }
    
    def subgroup_analysis(self):
        """Analyze model performance across demographic subgroups."""
        print("\nPerforming subgroup analysis...")
        
        # Define subgroups to analyze
        subgroup_configs = {
            'sex': {'column': 'sex', 'groups': {'Male': '1', 'Female': '2'}},
            'race': {'column': 'race7', 'groups': {
                'White': '1', 'Black': '2', 'Hispanic': '3', 
                'Asian': '4', 'Native American': '5', 'Other': ['6', '7']
            }},
            'education': {'column': 'educat', 'groups': {
                'High School or Less': ['1', '2', '3'], 
                'Some College': ['4', '5'], 
                'College Graduate+': ['6', '7']
            }},
            'smoking_status': {'column': 'cig_stat', 'groups': {
                'Never': '0', 'Former': '1', 'Current': '2'
            }},
            'nlst_eligible': {'column': 'nlst_flag', 'groups': {
                'NLST Eligible': '1', 'Not Eligible': ['0', '']
            }}
        }
        
        subgroup_results = {}
        
        for subgroup_name, config in subgroup_configs.items():
            print(f"\nAnalyzing {subgroup_name} subgroups...")
            subgroup_results[subgroup_name] = self._analyze_single_subgroup(
                subgroup_name, config['column'], config['groups']
            )
            
        return subgroup_results
    
    def _analyze_single_subgroup(self, subgroup_name, column, groups):
        """Analyze performance for a single subgroup category."""
        results = {}
        
        for group_name, group_values in groups.items():
            # Handle single value or list of values
            if isinstance(group_values, str):
                group_values = [group_values]
            
            # Find patients in this subgroup
            group_mask = np.array([
                r.get(column, '') in group_values for r in self.test_data
            ])
            
            if np.sum(group_mask) < 50:  # Skip small groups
                print(f"  Skipping {group_name}: only {np.sum(group_mask)} patients")
                continue
                
            # Calculate AUC for this subgroup
            group_labels = self.test_labels[group_mask]
            group_predictions = self.test_predictions[group_mask]
            
            if len(np.unique(group_labels)) < 2:  # Need both cancer and non-cancer cases
                print(f"  Skipping {group_name}: no cancer cases or all cancer cases")
                continue
                
            group_auc = roc_auc_score(group_labels, group_predictions)
            cancer_rate = np.mean(group_labels)
            
            results[group_name] = {
                'auc': group_auc,
                'n_patients': np.sum(group_mask),
                'n_cancer': np.sum(group_labels),
                'cancer_rate': cancer_rate
            }
            
            print(f"  {group_name}: AUC={group_auc:.3f}, N={np.sum(group_mask):,}, "
                  f"Cancer={np.sum(group_labels)} ({cancer_rate:.1%})")
        
        return results
    
    def plot_subgroup_rocs(self, subgroup_results):
        """Create ROC plots for each subgroup category."""
        print("\nGenerating subgroup ROC plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        subgroup_names = list(subgroup_results.keys())
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        
        for idx, (subgroup_name, results) in enumerate(subgroup_results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Plot ROC curve for each group in this subgroup
            for i, (group_name, stats) in enumerate(results.items()):
                # Get predictions for this group
                group_mask = np.array([
                    any(r.get(self._get_column_for_subgroup(subgroup_name), '') in 
                        self._get_values_for_group(subgroup_name, group_name) 
                        for _ in [None]) for r in self.test_data
                ])
                
                if np.sum(group_mask) < 50:
                    continue
                    
                group_labels = self.test_labels[group_mask]
                group_predictions = self.test_predictions[group_mask]
                
                if len(np.unique(group_labels)) < 2:
                    continue
                
                fpr, tpr, _ = roc_curve(group_labels, group_predictions)
                
                ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                       label=f'{group_name} (AUC={stats["auc"]:.3f}, N={stats["n_patients"]:,})')
            
            # Plot random classifier
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.6)
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curves by {subgroup_name.title()}')
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(subgroup_results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Subgroup Analysis: ROC Curves by Demographics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig("subgroup_roc_curves.png", dpi=300, bbox_inches="tight")
        plt.show()
    
    def _get_column_for_subgroup(self, subgroup_name):
        """Helper to get column name for subgroup."""
        mapping = {
            'sex': 'sex', 'race': 'race7', 'education': 'educat',
            'smoking_status': 'cig_stat', 'nlst_eligible': 'nlst_flag'
        }
        return mapping.get(subgroup_name, '')
    
    def _get_values_for_group(self, subgroup_name, group_name):
        """Helper to get values for a specific group."""
        # This is a simplified version - in practice you'd want to store the mapping
        return ['1'] if 'Male' in group_name else ['2']
    
    def analyze_disparities(self, subgroup_results):
        """Analyze disparities between subgroups."""
        print("\n" + "="*60)
        print("SUBGROUP DISPARITY ANALYSIS")
        print("="*60)
        
        disparity_findings = {}
        
        for subgroup_name, results in subgroup_results.items():
            if len(results) < 2:
                continue
                
            aucs = [stats['auc'] for stats in results.values()]
            group_names = list(results.keys())
            
            max_auc = max(aucs)
            min_auc = min(aucs)
            auc_range = max_auc - min_auc
            
            max_group = group_names[aucs.index(max_auc)]
            min_group = group_names[aucs.index(min_auc)]
            
            disparity_findings[subgroup_name] = {
                'max_auc': max_auc,
                'min_auc': min_auc,
                'range': auc_range,
                'max_group': max_group,
                'min_group': min_group,
                'concerning': auc_range > 0.05
            }
            
            print(f"\n{subgroup_name.upper()}:")
            print(f"  Best performing: {max_group} (AUC = {max_auc:.3f})")
            print(f"  Worst performing: {min_group} (AUC = {min_auc:.3f})")
            print(f"  Difference: {auc_range:.3f}")
            
            if auc_range > 0.05:
                print(f"  ⚠️  CONCERNING: >0.05 AUC difference detected")
            else:
                print(f"  ✅ Acceptable: <0.05 AUC difference")
        
        return disparity_findings

def main_comprehensive_analysis():
    """Run complete analysis: NLST + subgroups."""
    print("🎯 Comprehensive ROC Analysis: NLST + Subgroups")
    print("="*60)
    
    # Initialize and train model
    analyzer = ComprehensiveROCAnalyzer()
    analyzer.load_data()
    analyzer.train_best_model()
    analyzer.generate_test_predictions()
    
    # NLST analysis
    print("\n📊 PART 1: NLST Operating Point Analysis")
    nlst_stats = analyzer.analyze_nlst_operating_point()
    comparison_results = analyzer.plot_roc_with_nlst(nlst_stats)
    
    # Subgroup analysis
    print("\n📊 PART 2: Demographic Subgroup Analysis")
    subgroup_results = analyzer.subgroup_analysis()
    
    # Create simplified subgroup ROC plots
    analyzer.create_clean_subgroup_plots(subgroup_results)
    
    # Analyze disparities
    disparity_findings = analyzer.analyze_disparities(subgroup_results)
    
    # Generate summary report
    analyzer.generate_comprehensive_report(comparison_results, subgroup_results, disparity_findings)
    
    print(f"\n🎉 Complete Analysis Finished!")
    print(f"Generated files:")
    print(f"  • roc_curve_with_nlst.png")
    print(f"  • subgroup_roc_curves.png") 
    print(f"  • comprehensive_analysis_report.txt")
    print("="*60)
    
    return analyzer

def create_clean_subgroup_plots(self, subgroup_results):
    """Create clean, focused subgroup ROC plots."""
    print("\nGenerating clean subgroup ROC plots...")
    
    # Create a more focused plot with key subgroups
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Focus on key subgroups
    key_subgroups = ['sex', 'race', 'smoking_status', 'nlst_eligible']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, subgroup_name in enumerate(key_subgroups):
        if subgroup_name not in subgroup_results:
            continue
            
        ax = axes[idx]
        results = subgroup_results[subgroup_name]
        
        for i, (group_name, stats) in enumerate(results.items()):
            # Recreate ROC curve for this group
            group_mask = self._create_group_mask(subgroup_name, group_name)
            
            if np.sum(group_mask) < 50:
                continue
                
            group_labels = self.test_labels[group_mask]
            group_predictions = self.test_predictions[group_mask]
            
            if len(np.unique(group_labels)) < 2:
                continue
            
            fpr, tpr, _ = roc_curve(group_labels, group_predictions)
            
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5,
                   label=f'{group_name}\n(AUC={stats["auc"]:.3f}, N={stats["n_patients"]:,})')
        
        # Random classifier line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{subgroup_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Subgroup Analysis: Model Performance by Demographics', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("subgroup_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

def _create_group_mask(self, subgroup_name, group_name):
    """Helper to create mask for specific group."""
    column_mapping = {
        'sex': 'sex', 'race': 'race7', 'education': 'educat',
        'smoking_status': 'cig_stat', 'nlst_eligible': 'nlst_flag'
    }
    
    value_mapping = {
        ('sex', 'Male'): ['1'],
        ('sex', 'Female'): ['2'],
        ('smoking_status', 'Never'): ['0'],
        ('smoking_status', 'Former'): ['1'], 
        ('smoking_status', 'Current'): ['2'],
        ('nlst_eligible', 'NLST Eligible'): ['1'],
        ('nlst_eligible', 'Not Eligible'): ['0', '']
    }
    
    column = column_mapping.get(subgroup_name, '')
    values = value_mapping.get((subgroup_name, group_name), [])
    
    return np.array([r.get(column, '') in values for r in self.test_data])

# Add methods to the class
ComprehensiveROCAnalyzer.create_clean_subgroup_plots = create_clean_subgroup_plots
ComprehensiveROCAnalyzer._create_group_mask = _create_group_mask

if __name__ == "__main__":
    main_comprehensive_analysis()
