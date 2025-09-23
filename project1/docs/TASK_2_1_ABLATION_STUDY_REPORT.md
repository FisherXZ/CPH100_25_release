# Task 2.1: Ablation Study - What Makes Our Model Work?

## What We Did

We tested 10 different combinations of features to understand what makes our lung cancer prediction model so successful (83.9% accuracy). Think of it like testing a recipe - we tried different ingredients one by one to see which ones actually matter.

We started simple with just age, then gradually added more features like sex, smoking history, family history, and medical conditions. For each combination, we trained a new model and measured how well it predicted lung cancer risk.

## Key Results

**The Big Discovery: Smoking Features Are Everything**
- Age alone: 60.7% accuracy (our starting point)
- Age + pack-years (smoking intensity): 82.3% accuracy (+21.6 points!)
- Age + full smoking history: 83.9% accuracy (+23.2 points!)

**Best Model Features (6 total):**
- `age` - Patient age (normalized)
- `pack_years` - Smoking intensity over lifetime (packs per day × years smoked)
- `cig_years` - Total years of smoking
- `cigpd_f` - Cigarettes per day
- `smoked_f` - Ever smoked (yes/no)
- `cig_stat` - Current smoking status (never/former/current)

This was shocking - just adding one smoking variable (pack-years) gave us 92% of our total improvement. Everything else we tried barely moved the needle.

**Other Features Had Minimal Impact**
- Adding sex: +1.1% improvement (tiny)
- Adding demographics (race, education): +2.4% improvement (small)  
- Adding family history: +2.0% improvement (modest)
- Adding medical conditions: -0.6% improvement (actually made it worse!)

**Quality Beats Quantity**
Our best model uses only 6 features and beats our 30-feature model. More features don't always help - sometimes they just add noise.

## What This Means

**For Lung Cancer Screening:**
The most important thing to ask patients is detailed smoking history. Age and smoking variables capture almost all the predictive power we need. We don't need long, complicated questionnaires.

**For Our Model:**
- Pack-years (how much and how long someone smoked) is the single most important predictor
- Age + comprehensive smoking history gives us 83.9% accuracy with just 6 features
- Adding more medical history doesn't help and might hurt performance

**Why Our Model Works So Well:**
1. We focused on the right features (smoking history)
2. We used proper data preprocessing (normalization, encoding)
3. We found optimal training settings through grid search
4. We have enough data (100K patients) to train reliably

## Visual Results

We created comprehensive graphs showing our findings (saved as `ablation_study_performance.png`):

1. **Performance Comparison**: Bar chart showing all 10 experiments with the smoking models clearly dominating
2. **Feature Count vs Performance**: Scatter plot proving that more features ≠ better performance  
3. **Improvement Over Baseline**: Shows the massive +21.7 point jump when adding pack-years
4. **Category Summary**: Smoking features achieve 83.9% vs demographics at 63.0%

The graphs make it crystal clear: there's a huge performance cliff between smoking-based models (82-84% AUC) and everything else (60-63% AUC).

## The Bottom Line

Smoking history dominates lung cancer risk prediction. A simple model with age and detailed smoking information performs as well as complex models with many more variables. This makes sense clinically - smoking is the primary cause of lung cancer, so it should be the primary predictor.

Our ablation study proves that **smart feature selection matters more than having lots of features**. By focusing on what actually drives lung cancer risk (smoking), we built a model that's both accurate and practical for real-world screening programs.

**Key Numbers:**
- **93% of our model's improvement** comes from smoking features
- **6 features perform better** than 30 features  
- **Pack-years alone** provides +21.7 percentage points improvement
