"""
Model Comparison: 6-Feature vs 15-Feature Configuration
Compare the performance of smoking-focused vs comprehensive feature sets
"""

import numpy as np
from csv import DictReader
import random
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from vectorizer import Vectorizer
from logistic_regression import LogisticRegression

def compare_models():
    """Compare 6-feature smoking model vs 15-feature comprehensive model."""
    print("🔍 Model Feature Comparison: 6-Feature vs 15-Feature")
    print("="*60)
    
    # Load and split data
    reader = DictReader(open("lung_prsn.csv", "r"))
    rows = [r for r in reader]
    
    NUM_TRAIN, NUM_VAL = 100000, 25000
    random.seed(0)
    random.shuffle(rows)
    
    train_data = rows[:NUM_TRAIN]
    val_data = rows[NUM_TRAIN:NUM_TRAIN+NUM_VAL]
    test_data = rows[NUM_TRAIN+NUM_VAL:]
    
    print(f"Data split: {len(train_data):,} train, {len(val_data):,} val, {len(test_data):,} test")
    
    # Define feature configurations
    smoking_6_config = {
        "numerical": ["age", "pack_years", "cig_years", "cigpd_f"],
        "categorical": ["smoked_f"],
        "ordinal": ["cig_stat"]
    }
    
    comprehensive_15_config = {
        "numerical": ["age", "bmi_curr", "pack_years", "cig_years", "cigpd_f"],
        "categorical": ["sex", "race7", "fh_cancer", "lung_fh", "smoked_f", 
                       "diabetes_f", "hearta_f", "stroke_f"],
        "ordinal": ["educat", "cig_stat"]
    }
    
    # Optimal hyperparameters
    hyperparams = {
        'num_epochs': 200,
        'learning_rate': 0.001,
        'batch_size': 128,
        'regularization_lambda': 0
    }
    
    models = {}
    results = {}
    
    # Train both models
    for model_name, feature_config in [
        ("6-Feature Smoking", smoking_6_config),
        ("15-Feature Comprehensive", comprehensive_15_config)
    ]:
        print(f"\n🚀 Training {model_name} Model...")
        print(f"Features: {sum(len(v) for v in feature_config.values())} total")
        
        # Initialize model and vectorizer
        vectorizer = Vectorizer(feature_config)
        model = LogisticRegression(
            num_epochs=hyperparams['num_epochs'],
            learning_rate=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            regularization_lambda=hyperparams['regularization_lambda'],
            verbose=False
        )
        
        # Prepare data
        vectorizer.fit(train_data)
        train_X = vectorizer.transform(train_data)
        val_X = vectorizer.transform(val_data)
        test_X = vectorizer.transform(test_data)
        
        train_Y = np.array([int(r["lung_cancer"]) for r in train_data])
        val_Y = np.array([int(r["lung_cancer"]) for r in val_data])
        test_Y = np.array([int(r["lung_cancer"]) for r in test_data])
        
        # Train model
        model.fit(train_X, train_Y, val_X, val_Y)
        
        # Evaluate
        train_pred = model.predict_proba(train_X)
        val_pred = model.predict_proba(val_X)
        test_pred = model.predict_proba(test_X)
        
        train_auc = roc_auc_score(train_Y, train_pred)
        val_auc = roc_auc_score(val_Y, val_pred)
        test_auc = roc_auc_score(test_Y, test_pred)
        
        # Store results
        models[model_name] = {
            'model': model,
            'vectorizer': vectorizer,
            'predictions': test_pred,
            'labels': test_Y
        }
        
        results[model_name] = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'test_auc': test_auc,
            'features': feature_config,
            'feature_count': sum(len(v) for v in feature_config.values())
        }
        
        print(f"  Train AUC: {train_auc:.4f}")
        print(f"  Val AUC:   {val_auc:.4f}")
        print(f"  Test AUC:  {test_auc:.4f}")
    
    # Performance comparison
    print(f"\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    
    smoking_results = results["6-Feature Smoking"]
    comprehensive_results = results["15-Feature Comprehensive"]
    
    print(f"6-Feature Smoking Model:")
    print(f"  Features: {smoking_results['feature_count']}")
    print(f"  Test AUC: {smoking_results['test_auc']:.4f}")
    
    print(f"\n15-Feature Comprehensive Model:")
    print(f"  Features: {comprehensive_results['feature_count']}")
    print(f"  Test AUC: {comprehensive_results['test_auc']:.4f}")
    
    auc_difference = comprehensive_results['test_auc'] - smoking_results['test_auc']
    print(f"\nAUC Difference: {auc_difference:+.4f}")
    
    if abs(auc_difference) < 0.005:
        print("✅ Models perform essentially the same (difference < 0.5%)")
    elif auc_difference > 0:
        print(f"📈 15-Feature model performs better (+{auc_difference:.1%})")
    else:
        print(f"📉 6-Feature model performs better (+{abs(auc_difference):.1%})")
    
    # Feature efficiency analysis
    feature_efficiency_6 = smoking_results['test_auc'] / smoking_results['feature_count']
    feature_efficiency_15 = comprehensive_results['test_auc'] / comprehensive_results['feature_count']
    
    print(f"\n📏 Feature Efficiency (AUC per feature):")
    print(f"  6-Feature:  {feature_efficiency_6:.4f}")
    print(f"  15-Feature: {feature_efficiency_15:.4f}")
    
    if feature_efficiency_6 > feature_efficiency_15:
        efficiency_advantage = ((feature_efficiency_6 / feature_efficiency_15) - 1) * 100
        print(f"✅ 6-Feature model is {efficiency_advantage:.1f}% more efficient")
    else:
        efficiency_advantage = ((feature_efficiency_15 / feature_efficiency_6) - 1) * 100
        print(f"📊 15-Feature model is {efficiency_advantage:.1f}% more efficient")
    
    # Create ROC comparison plot
    print(f"\n📈 Generating ROC comparison plot...")
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#2E8B57', '#DC143C']  # Sea green, Crimson
    
    for i, (model_name, model_data) in enumerate(models.items()):
        predictions = model_data['predictions']
        labels = model_data['labels']
        test_auc = results[model_name]['test_auc']
        
        fpr, tpr, _ = roc_curve(labels, predictions)
        
        plt.plot(fpr, tpr, color=colors[i], lw=3,
                label=f'{model_name} (AUC = {test_auc:.3f})')
    
    # Random classifier line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison: 6-Feature vs 15-Feature Models', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add performance summary text box
    textstr = f'''Model Comparison Summary:
6-Feature:  {smoking_results['test_auc']:.4f} AUC ({smoking_results['feature_count']} features)
15-Feature: {comprehensive_results['test_auc']:.4f} AUC ({comprehensive_results['feature_count']} features)
Difference: {auc_difference:+.4f} AUC'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig("model_feature_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Recommendation
    print(f"\n" + "="*60)
    print("🎯 RECOMMENDATION")
    print("="*60)
    
    if abs(auc_difference) < 0.01:  # Less than 1% difference
        print("✅ CHOOSE 6-FEATURE MODEL")
        print("Reasons:")
        print("  • Equivalent performance with 60% fewer features")
        print("  • Simpler data collection and maintenance")
        print("  • Reduced risk of overfitting")
        print("  • Better interpretability and clinical adoption")
        print("  • Lower computational requirements")
    elif smoking_results['test_auc'] > comprehensive_results['test_auc']:
        print("✅ CHOOSE 6-FEATURE MODEL")
        print("Reasons:")
        print("  • Better performance with fewer features")
        print("  • Superior feature efficiency")
        print("  • Simpler implementation")
    else:
        improvement = auc_difference * 100
        if improvement < 2:  # Less than 2% improvement
            print("✅ CHOOSE 6-FEATURE MODEL")
            print(f"Reasons:")
            print(f"  • Only {improvement:.1f}% performance gain from 9 extra features")
            print(f"  • Diminishing returns on feature complexity")
            print(f"  • Better practical deployment characteristics")
        else:
            print("📊 CONSIDER 15-FEATURE MODEL")
            print(f"Reasons:")
            print(f"  • {improvement:.1f}% performance improvement")
            print(f"  • May be worth the added complexity for clinical applications")
    
    print(f"\n🎉 Comparison Complete!")
    print(f"Generated: model_feature_comparison.png")
    
    return results

if __name__ == "__main__":
    compare_models()
