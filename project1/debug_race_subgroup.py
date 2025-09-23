"""
Debug Race Subgroup Analysis
Investigate why race subgroup shows no data in ROC analysis
"""

import numpy as np
from csv import DictReader
import random
from collections import Counter

def debug_race_data():
    """Debug race column data distribution and quality."""
    print("🔍 Debugging Race Subgroup Data")
    print("="*50)
    
    # Load data with same split
    reader = DictReader(open("lung_prsn.csv", "r"))
    rows = [r for r in reader]
    
    NUM_TRAIN, NUM_VAL = 100000, 25000
    random.seed(0)
    random.shuffle(rows)
    
    test_data = rows[NUM_TRAIN+NUM_VAL:]
    print(f"Test set size: {len(test_data):,} patients")
    
    # Examine race7 column
    print(f"\n📊 RACE7 COLUMN ANALYSIS:")
    race_values = [r.get('race7', '') for r in test_data]
    race_counts = Counter(race_values)
    
    print(f"All race7 values found:")
    for value, count in race_counts.most_common():
        percentage = (count / len(test_data)) * 100
        print(f"  '{value}': {count:,} patients ({percentage:.1f}%)")
    
    # Check for missing/empty values
    empty_count = sum(1 for r in race_values if r in ['', None, 'nan', 'NaN'])
    print(f"\nMissing/empty values: {empty_count:,} ({empty_count/len(test_data)*100:.1f}%)")
    
    # Test our grouping logic
    print(f"\n🧪 TESTING GROUPING LOGIC:")
    
    race_groups = {
        'White': lambda r: r.get('race7', '') == '1',
        'Black': lambda r: r.get('race7', '') == '2', 
        'Hispanic': lambda r: r.get('race7', '') == '3',
        'Asian': lambda r: r.get('race7', '') == '4',
        'Native American': lambda r: r.get('race7', '') == '5',
        'Other': lambda r: r.get('race7', '') in ['6', '7']
    }
    
    for group_name, condition in race_groups.items():
        mask = np.array([condition(r) for r in test_data])
        group_size = np.sum(mask)
        percentage = (group_size / len(test_data)) * 100
        print(f"  {group_name}: {group_size:,} patients ({percentage:.1f}%)")
        
        # Check cancer cases in this group
        if group_size > 0:
            cancer_cases = sum(1 for i, r in enumerate(test_data) 
                             if mask[i] and r.get('lung_cancer', '') == '1')
            cancer_rate = (cancer_cases / group_size) * 100 if group_size > 0 else 0
            print(f"    Cancer cases: {cancer_cases} ({cancer_rate:.1f}%)")
    
    # Check what our minimum threshold was
    print(f"\n⚠️  MINIMUM THRESHOLD ANALYSIS:")
    print(f"Groups with <100 patients (our threshold):")
    for group_name, condition in race_groups.items():
        mask = np.array([condition(r) for r in test_data])
        group_size = np.sum(mask)
        if group_size < 100:
            print(f"  {group_name}: {group_size} patients (EXCLUDED)")
        else:
            print(f"  {group_name}: {group_size} patients (INCLUDED)")
    
    # Check for data quality issues
    print(f"\n🔍 DATA QUALITY CHECK:")
    unique_values = set(race_values)
    print(f"Unique race7 values: {sorted(unique_values)}")
    
    # Look for unexpected values
    expected_values = {'1', '2', '3', '4', '5', '6', '7', ''}
    unexpected = unique_values - expected_values
    if unexpected:
        print(f"⚠️  Unexpected values found: {unexpected}")
    else:
        print(f"✅ All values are expected")
    
    # Test with lower threshold
    print(f"\n📈 TESTING WITH LOWER THRESHOLD (50 patients):")
    for group_name, condition in race_groups.items():
        mask = np.array([condition(r) for r in test_data])
        group_size = np.sum(mask)
        if group_size >= 50:
            # Check if we have both cancer and non-cancer cases
            cancer_cases = sum(1 for i, r in enumerate(test_data) 
                             if mask[i] and r.get('lung_cancer', '') == '1')
            non_cancer = group_size - cancer_cases
            
            print(f"  {group_name}: {group_size} total, {cancer_cases} cancer, {non_cancer} non-cancer")
            
            if cancer_cases > 0 and non_cancer > 0:
                print(f"    ✅ Suitable for ROC analysis")
            else:
                print(f"    ❌ No variation in outcomes")

if __name__ == "__main__":
    debug_race_data()
