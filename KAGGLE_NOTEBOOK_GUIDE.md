# Kaggle Notebook - Updates & Fixes

Updated: February 17, 2026

## 🔧 What Was Fixed

### 1. **Data Path Issues**

- ✅ Fixed incorrect path: `sequences.pkl` → `ml1m_sequential.pkl`
- ✅ Updated all experiment scripts to use correct data file

### 2. **Results Display & Saving Errors**

- ✅ Fixed metric key case sensitivity (`ndcg@10` → `NDCG@10`)
- ✅ Added error handling for missing results files
- ✅ Fixed grouped metrics display (handles missing 'long' group gracefully)
- ✅ Improved JSON parsing with try/catch blocks

### 3. **Analysis Script Fixes**

- ✅ `analyze_results.py` now uses correct capitalized metric keys
- ✅ Added graceful handling of missing data
- ✅ Fixed CSV export functionality

### 4. **Enhanced Features**

- ✅ Added priority experiment guide
- ✅ Added alpha value tracking and display
- ✅ Added grid search for optimal alpha (commented out)
- ✅ Added all variants training section
- ✅ Improved error messages and user feedback
- ✅ Better progress indicators

---

## 📊 Notebook Structure

### Quick Start (15-20 min with GPU)

1. **Clone & Setup** - Get repository and dependencies
2. **Verify GPU** - Check hardware availability
3. **Quick Test** - 2-epoch sanity check
4. **Train Hybrid** - Main model (50 epochs)
5. **Train Baseline** - SASRec comparison
6. **Analyze** - View results

### Advanced Experiments (Optional, 8-12 hours)

7. **Grid Search Alpha** - Find optimal fusion weight
8. **All Variants** - Complete ablation study

### Results & Download

9. **Display Results** - Tables and comparisons
10. **User Groups** - Performance by history length
11. **Alpha Stats** - Fusion weight analysis
12. **Learning Curves** - Training visualization
13. **Download** - Package results for local analysis

---

## 🚀 How to Use on Kaggle

### 1. Create New Notebook

1. Go to Kaggle.com
2. Click "+ New Notebook"
3. Change kernel type: Python → Notebook

### 2. Upload Notebook File

1. File → Import Notebook
2. Select `kaggle_notebook.ipynb`
3. Or copy-paste cells manually

### 3. Enable GPU (Recommended)

1. Settings (right panel) → Accelerator
2. Select: GPU T4 x2 (free tier)
3. Save

### 4. Set Internet Access

1. Settings → Internet → On
2. This allows git clone

### 5. Run Notebook

- **Option A:** Run All (Cell → Run All)
- **Option B:** Run cells one by one (Shift+Enter)

### 6. Wait for Training

- Hybrid Discrete: ~8-10 minutes (GPU)
- SASRec Baseline: ~8-10 minutes (GPU)
- Total: ~20 minutes for both

### 7. Download Results

- After completion, scroll to bottom
- Download `results.zip` from Output tab
- Extract locally and run analysis scripts

---

## 🎯 Priority Experiments Queue

### Must Run (Priority 1)

```python
# 1. SASRec Baseline
!python experiments/run_experiment.py --model sasrec --epochs 50 --patience 10

# 2. Hybrid Discrete (our best)
!python experiments/run_experiment.py --model hybrid_discrete --epochs 50 --patience 10
```

**Time:** ~20 minutes with GPU  
**Goal:** Verify we beat baseline

### Should Run (Priority 2)

```python
# 3. Hybrid Fixed (best current performer)
!python experiments/run_experiment.py --model hybrid_fixed --epochs 50 --patience 10

# 4. Hybrid Learnable
!python experiments/run_experiment.py --model hybrid_learnable --epochs 50 --patience 10

# 5. Hybrid Continuous
!python experiments/run_experiment.py --model hybrid_continuous --epochs 50 --patience 10
```

**Time:** ~30 minutes with GPU  
**Goal:** Complete comparison

### Advanced (Priority 3 - If Time Permits)

```python
# 6. Grid Search for Optimal Alpha
for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
    !python experiments/run_experiment.py \
        --model hybrid_fixed \
        --fixed_alpha {alpha} \
        --epochs 50
```

**Time:** ~2-3 hours with GPU  
**Goal:** Find optimal fusion weight

---

## 📝 Common Issues & Solutions

### Issue: "No module named 'src'"

**Solution:** Make sure you're in the `length-adaptive` directory:

```python
%cd length-adaptive
```

### Issue: "FileNotFoundError: sequences.pkl"

**Solution:** Already fixed! The notebook now uses `ml1m_sequential.pkl`

### Issue: "KeyError: 'ndcg@10'"

**Solution:** Already fixed! Now uses correct capitalization `NDCG@10`

### Issue: "No results found"

**Solution:** Run experiments first (Steps 5-6) before analysis (Steps 8-12)

### Issue: GPU not available

**Solution:**

1. Settings → Accelerator → GPU T4
2. If quota exceeded, wait or use CPU (slower: ~40 min per model)

### Issue: "long" group missing in results

**Cause:** Dataset may have very few users with >50 interactions  
**Solution:** This is normal - results will show Short & Medium groups

---

## 📦 Output Files Explained

After running experiments, you'll have:

```
results/
├── hybrid_discrete_YYYYMMDD_HHMMSS/
│   ├── best_model.pt           # Best model checkpoint
│   ├── checkpoint.pt            # Latest checkpoint
│   ├── config.json              # Experiment configuration
│   ├── history.json             # Training history (loss, val metrics)
│   └── results.json             # Final test metrics ⭐
│
├── sasrec_YYYYMMDD_HHMMSS/
│   └── ... (same structure)
│
├── overall_comparison.csv       # Generated by analyze_results.py
├── comparison_short.csv         # Short-history users
├── comparison_medium.csv        # Medium-history users
└── learning_curves.png          # Visualization
```

### Most Important File: `results.json`

```json
{
  "test_metrics": {
    "HR@10": 0.0999,    // Hit Rate @ 10
    "NDCG@10": 0.0471,  // Main metric ⭐
    "MRR@10": 0.0314    // Mean Reciprocal Rank
  },
  "grouped_metrics": {
    "short": { ... },   // Users with ≤10 items
    "medium": { ... },  // Users with 11-50 items
    "overall": { ... }
  },
  "best_epoch": 39,
  "best_val_metric": 0.0510
}
```

---

## 🔍 Verifying Success

After training both models, check:

### 1. Training Completed Successfully

```python
!ls results/
```

Should show folders like:

- `sasrec_20260217_XXXXXX/`
- `hybrid_discrete_20260217_XXXXXX/`

### 2. Results Files Exist

```python
!ls results/*/results.json
```

Should list JSON files for each experiment

### 3. Performance Metrics

Run cell 9 to see comparison table. Look for:

- ✅ Hybrid NDCG@10 > SASRec NDCG@10
- ✅ Improvement > +2%

### 4. Download Successful

```python
!ls -lh results.zip
```

Should show ~50-100MB file

---

## 🎓 For Your Teacher

After downloading results.zip, extract it and run local analysis:

```bash
cd /path/to/extracted/results
cd ..  # Go to project root

# Activate environment
source venv/bin/activate

# Run comprehensive analysis
bash experiments/run_all_analysis.sh
```

This generates:

- Comparison tables
- Statistical significance tests
- Visualizations
- User distribution analysis

---

## ⏱️ Time Estimates

| Task                   | GPU T4      | CPU         |
| ---------------------- | ----------- | ----------- |
| Clone & Setup          | 2 min       | 2 min       |
| Quick Test (2 epochs)  | 1 min       | 5 min       |
| SASRec (50 epochs)     | 8 min       | 40 min      |
| Hybrid (50 epochs)     | 10 min      | 45 min      |
| Analysis & Viz         | 1 min       | 1 min       |
| **Total (Priority 1)** | **~20 min** | **~90 min** |
| All 5 models           | ~40 min     | ~3-4 hours  |
| Grid search (5 alphas) | ~2 hours    | ~8 hours    |

---

## 📚 Additional Resources

- **Action Plan:** `ACTION_PLAN.md` - Complete strategy
- **Kaggle Experiments:** `KAGGLE_EXPERIMENTS_TODO.md` - Detailed experiments
- **Current Status:** `CURRENT_STATUS.md` - Quick reference
- **Analysis Guide:** `experiments/README.md` - Local analysis tools

---

## ✅ Checklist

Before submitting to teacher:

- [ ] Trained SASRec baseline
- [ ] Trained at least one hybrid model
- [ ] Verified hybrid beats baseline
- [ ] Downloaded results.zip
- [ ] Extracted locally
- [ ] Ran local analysis scripts
- [ ] Generated visualizations
- [ ] Created comparison tables
- [ ] Documented findings

---

**Need Help?**

- Check `CURRENT_STATUS.md` for quick commands
- See `ACTION_PLAN.md` for detailed strategy
- Review error messages carefully - most are self-explanatory

**Last Updated:** February 17, 2026
