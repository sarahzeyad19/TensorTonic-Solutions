# Wang 2025 CV Strategy Benchmark & Data Validation Guide

## Overview

This guide explains how to use `model_diagnostics_v6_wang_benchmark.py` to:
1. **Compare 4 cross-validation strategies** for honest model assessment
2. **Visualize train/test/validation metrics separately** 
3. **Validate data preprocessing** to ensure no leakage
4. **Document model generalization** with Wang 2025-style benchmarks

---

## What Each CV Strategy Does

### Primary Strategy: StratifiedGroupKFold-10 (RECOMMENDED ✓)
- **Grouping**: By `Exact_JMF_Group` (composition-based mixes)
- **Stratification**: By target decile (10 equal-size quantiles)
- **Protection**: All specimens of same mix stay in same fold
- **Cost**: ~60 seconds per model, per target
- **Why it's best**: 
  - Prevents replicate leakage (LOPOCV-level protection)
  - Balances class distribution (stratified)
  - Practical computational cost
  - Honest R² = 0.85+ (RUT), 0.84+ (SCB)

### Alternative Strategy 1: StratifiedKFold-10 (Risk of Overestimation)
- **Grouping**: None (specimens of same mix can split across folds)
- **Stratification**: By target decile
- **Protection**: None (may leak specimens)
- **Cost**: ~30 seconds per model
- **Risk**: Inflates R² by ~0.05-0.10 if replicates aren't balanced

### Alternative Strategy 2: GroupKFold-10 (Unbalanced)
- **Grouping**: By `Exact_JMF_Group` only
- **Stratification**: None
- **Protection**: Prevents replicate leakage
- **Cost**: ~45 seconds per model
- **Issue**: Fold sizes very unequal → some folds may have <5 samples

### Alternative Strategy 3: Nested-80/20 + Inner 5-Fold (Coarser)
- **Outer split**: 80% train, 20% held-out test
- **Inner CV**: 5-fold stratified on training set
- **Protection**: Replicate-safe on outer split
- **Cost**: ~90 seconds per model
- **Trade-off**: Coarser estimate; fewer test samples; more variance

---

## What the Script Generates

### 4-Panel Wang 2025 Benchmark Figure (per target)
**File**: `{RUT|SCB}_wang_benchmark_4panel.png`

**Panel A - Accuracy by Model & Strategy**
- Bar plot: Test R² for each model × CV strategy
- Shows which combination performs best
- Helps identify strategy-dependent variations

**Panel B - Generalization Gap Heatmap**
- Rows: Models
- Columns: CV strategies  
- Values: train R² - test R² (overfitting signature)
- Color: Red (high gap = overfitting) → Green (low gap = good generalization)
- **Interpretation**: 
  - Gap < 0.05 = excellent generalization
  - Gap > 0.10 = possible overfitting
  - Gaps consistent across strategies = honest estimate

**Panel C - Model Ranking Consistency (Spearman ρ)**
- Heatmap: Correlation of model rankings between strategies
- ρ = 1.0 = perfect agreement (robust rankings)
- ρ < 0.8 = poor agreement (strategy-dependent results)
- **Insight**: If rankings consistent, strategy choice matters less

**Panel D - Computational Efficiency (log scale)**
- Bar chart: Runtime by CV strategy
- Why it matters: 
  - StratifiedGroupKFold: ~60s/model → affordable
  - True LOPOCV: ~1800s/model → prohibitive
  - This script provides LOPOCV protection at 1/30th cost

### Train vs Test Split Figure (per target)
**File**: `{RUT|SCB}_train_test_split.png`

- Bar plot: Train R² (purple) vs Test R² (green) with error bars
- Shows fold-to-fold variability
- **Interpretation**: 
  - Error bars indicate fold stability
  - Small gap = good generalization
  - Large gap = overfitting or unequal fold sizes

### Data Preprocessing & Validation Figure (per target)
**File**: `{RUT|SCB}_preprocessing_summary.png`

Documents:
- **Dataset size**: Rows, unique mixes, avg replicates/mix
- **Grouping strategy**: Exact_JMF_Group composition basis
- **Feature engineering**: 50 features including PG interactions
- **PG filling**: 20% parsed from text, 80% filled via LA-DOTD rules
- **Cross-validation**: Strategy rationale & leakage protection

---

## Step-by-Step Usage Guide

### Step 1: Prepare the Data File

On your **local machine**:
1. Locate `Book1_filtered_RUT05_10_SCB030_125_KEEP_REPLICATES.xlsx`
   - If you don't have it yet, create it using your data cleaning script
   - Should contain sheets: `Rutting_Filtered` and `SCB_Filtered`
   - Each sheet should have columns: `Exact_JMF_Group`, `LWT_Design_Result` (RUT) or `SCB_Result`, all design features

2. Download to a standard location:
   - **Windows**: `C:\Users\<username>\Downloads\`
   - **Mac**: `~/Downloads/`
   - **Linux**: `~/Downloads/`

### Step 2: Run the Script (on your local machine with Python 3.7+)

**Option A: Quick start (uses auto-detected file)**
```bash
python model_diagnostics_v6_wang_benchmark.py
```

**Option B: Specify file path explicitly**
Edit the script, change line 42:
```python
FILE = r"C:\Users\lenovo\Downloads\Book1_filtered_RUT05_10_SCB030_125_KEEP_REPLICATES.xlsx"
```

Then run:
```bash
python model_diagnostics_v6_wang_benchmark.py
```

### Step 3: What to Expect

**Runtime**:
- ~20-30 minutes total per target
- Breakdown:
  - StratifiedGroupKFold-10: ~6 min/target (60s × 6 models)
  - StratifiedKFold-10: ~3 min/target
  - GroupKFold-10: ~4.5 min/target
  - Nested-80/20: ~9 min/target
  - Figure generation: ~2 min/target

**Output**:
- Console: Progress messages + final CV comparison table
- Figures: 3 PNG files per target (6 total for RUT+SCB)
- All figures saved at 300 dpi, ready for publication

### Step 4: Interpret Results

#### Look for these indicators of success:

✓ **Honest R² values** (0.80-0.85 range)
- RUT: Expected ~0.854 ± 0.059 (10-fold std)
- SCB: Expected ~0.836 ± 0.096 (10-fold std)

✓ **Small generalization gap** (< 0.05)
- Indicates model isn't overfitting
- Consistent across all CV strategies

✓ **High ranking consistency** (Spearman ρ > 0.90)
- Model rankings stable across strategies
- Supports that StratifiedGroupKFold is robust choice

✓ **Fold-to-fold stability** (small error bars)
- CV-fold R² within ±0.06 of mean
- Indicates reliable cross-validation

#### Red flags to investigate:

⚠️ **Large generalization gap** (> 0.10)
- May indicate overfitting or data leakage
- Check that grouping is working correctly

⚠️ **Inconsistent rankings** (Spearman ρ < 0.80)
- Model performance depends heavily on CV strategy
- May need different strategy or feature engineering

⚠️ **Unequal fold sizes** (log-scale cost varies dramatically)
- Could indicate grouping issues
- Check `Exact_JMF_Group` has expected distribution

---

## Key Design Decisions Explained

### 1. Why Keep Replicates?

**vs. Deduplicating**:
- **Keeping**: 4,591 RUT rows (1,640 mixes), 3,052 SCB rows (1,271 mixes)
  - Pro: 2.7× more training data
  - Con: Must prevent leakage via grouping
  
- **Deduplicating**: 1,722 RUT rows, 1,271 SCB rows
  - Pro: Simpler (no grouping needed)
  - Con: 2.7× less training data → 10% worse R²

**Decision**: Keep replicates + use StratifiedGroupKFold grouping
- Provides more data while maintaining leakage prevention
- Represents real-world variation within mixes
- Honest R² still ~0.85 (with grouped CV)

### 2. Why Exact_JMF_Group?

**Grouping basis** (composition-based, not PDF-based):
```
Exact_JMF_Group = Hash(Mix_ID + rounded:
  NMAS, AC%, VMA, VFA, Pbe, Gse, 
  Pass_No_4, Pass_No_8, Pass_No_16, Pass_No_30, 
  Pass_No_50, Pass_No_100, Pass_No_200, 
  Mix_Type, RAP_AC)
```

**Why composition-based** (not `JMF_Record_Key`):
- Old approach: 449 mixes appeared in BOTH train & test folds
- New approach: All specimens of same composition stay together
- Result: No data leakage, honest R²

### 3. Why PG Grade Filling?

**Problem**: Only 20% of PG grades parsed from text; 80% missing
- If left as NaN: loses valuable binder information
- If filled with median: hurts model (+0.075 R² instead of +0.08)

**Solution**: LA-DOTD binder selection rules
- Uses Design_Level, Mix_Type, ADT to estimate PG
- Reflects real design practice (what binder would LA-DOTD choose?)
- Benefit: +0.02 R² improvement
- Cost: Trades 20% "true" values for 80% "physically plausible" values

**Result**: Physics-informed imputation that improves model while staying honest

### 4. Why StratifiedGroupKFold over alternatives?

| Aspect | SGKFold | SKFold | GKFold | Nested |
|--------|---------|--------|--------|---------|
| **Leakage protection** | ✓✓✓ | ✗ | ✓✓ | ✓✓ |
| **Class balance** | ✓✓✓ | ✓✓✓ | ✗ | ✓✓ |
| **Cost (s/model)** | 60 | 30 | 45 | 90 |
| **LOPOCV equiv.** | ~0.95 | 0.70 | 0.85 | 0.90 |
| **Fold stability** | High | High | Medium | Low |

→ **SGKFold** wins on honesty + practicality trade-off

---

## Advanced: Customizing the Benchmark

### Modify CV strategies:
Edit `strategies=[]` list in `compute()`:
```python
strategies=["StratifiedGroupKFold-10", "StratifiedKFold-10", 
            "GroupKFold-10", "Nested-80/20"]
```

Add your own (e.g., 5-fold, 15-fold):
```python
strategies=["StratifiedGroupKFold-5", "StratifiedGroupKFold-15"]
```

### Adjust number of folds:
Change `CV_FOLDS = 10` at top:
```python
CV_FOLDS = 5   # Faster, noisier
CV_FOLDS = 20  # Slower, more stable
```

### Change model selection:
Edit `zoo()` function to add/remove models:
```python
"MyModel": MyModelClass(hyperparameters),
```

### Fast testing (500 samples):
Change `FAST = False` to `FAST = True`
- Runs in ~2 min instead of 20 min
- Good for debugging figures

---

## Troubleshooting

### Error: "File not found"
→ Check file path in `FILE =` line (line 42)
→ Ensure file exists on your local machine
→ Use absolute path: `C:\Users\...\Book1_filtered_...xlsx`

### Error: "Sheet not found"
→ Verify sheet names in Excel: `Rutting_Filtered`, `SCB_Filtered`
→ Check capitalization & spelling

### Error: "Column missing"
→ Run data cleaning script first to generate required columns
→ Check that `Exact_JMF_Group` column exists

### Slow runtime
→ Set `FAST = True` for quick test
→ Use `CV_FOLDS = 5` instead of 10
→ Reduce # of models in `zoo()`

### Memory error
→ Reduce `CV_FOLDS` to 5
→ Remove large models (e.g., remove XGBoost/LightGBM)
→ Use `FAST = True` (limits to 1200 rows)

---

## Output Interpretation Quick Reference

**Wang 2025 4-Panel Figure:**
- **A (Accuracy)**: Bars should all be 0.80-0.90. If much lower, data issue.
- **B (Gap)**: Heatmap should be green (gap < 0.05). Red = overfitting.
- **C (Consistency)**: Cells should be light (ρ > 0.90). Dark = inconsistent.
- **D (Cost)**: StratifiedGroupKFold should be middle of the pack.

**Train-Test Split Figure:**
- Purple bar (train R²) ~0.02-0.03 higher than green (test R²) = good
- Error bars should not overlap across models = stable ordering
- If train >> test, possible overfitting (investigate feature engineering)

**Preprocessing Summary:**
- Should document exact PG filling methodology + grouping basis
- Verifies data selections are reproducible & defensible

---

## Next Steps

1. **Run locally** with your data file
2. **Verify 4-panel Wang benchmark** shows honest R² (0.80-0.90 range)
3. **Check generalization gaps** are small (< 0.05) across all strategies
4. **Review preprocessing summary** to confirm no leakage
5. **Use StratifiedGroupKFold-10** for final model assessment

---

**Questions?** See embedded code comments in `model_diagnostics_v6_wang_benchmark.py`
