# Length-Adaptive Hybrid GNN + SASRec

## Analysis & Improvement Plan

**Status:** ✅ Successfully beating SASRec baseline  
**Best Model:** Hybrid Fixed (α=0.5)  
**Overall Improvement:** +3.7% HR@10  
**Short-User Improvement:** +42.2% HR@10

---

## 📋 What We've Built

### ✅ Completed Components

1. **Action Plan** ([ACTION_PLAN.md](ACTION_PLAN.md))
   - Complete strategy to beat SASRec
   - Timeline through March 1st
   - Success criteria and metrics

2. **Analysis Tools** ([experiments/](experiments/))
   - User distribution analyzer
   - Quick results comparison
   - Statistical significance tests
   - Visualization generator
   - Comprehensive analysis pipeline

3. **Kaggle Experiments TODO** ([KAGGLE_EXPERIMENTS_TODO.md](KAGGLE_EXPERIMENTS_TODO.md))
   - Priority experiments to run
   - Code improvements needed
   - Expected performance gains
   - Testing strategy

4. **Enhanced Evaluator** ([src/eval/evaluator.py](src/eval/evaluator.py))
   - Alpha value tracking
   - Long-history user metrics
   - Statistical analysis support

---

## 🚀 Quick Start

### For Local Analysis

```bash
# Setup environment (first time only)
bash experiments/setup_analysis_env.sh

# Then activate venv
source venv/bin/activate

# Run quick comparison
python experiments/quick_compare.py

# Run all analyses
bash experiments/run_all_analysis.sh

# Create visualizations
python experiments/create_visualizations.py
```

### For Kaggle Training

**⚠️ Important:** You already have SASRec baseline results! No need to retrain.

#### Smart Approach (Recommended):

1. Open `kaggle_notebook.ipynb` and upload to Kaggle
2. **Skip Step 6** (SASRec training) - Use existing baseline
3. **Run Step 5** (Hybrid training) - Train new models only
4. Download new results and merge with existing results/
5. Run local analysis scripts to compare everything

#### When to Retrain SASRec:

- ❌ Don't retrain if: Same data, same hyperparameters
- ✅ Do retrain if: Changed preprocessing, changed hyperparameters, need reproducibility check

**Time Saved:** ~8-10 minutes per experiment by skipping unchanged baseline!

---

## 📊 Current Results Summary

| Model             | HR@10 Overall | HR@10 Short   | Improvement |
| ----------------- | ------------- | ------------- | ----------- |
| SASRec (baseline) | 9.63%         | 11.73%        | —           |
| **Hybrid Fixed**  | **9.99%** ✅  | **16.67%** ✅ | **+42.2%**  |
| Hybrid Continuous | 9.61%         | 15.43%        | +31.5%      |
| Hybrid Learnable  | 9.33%         | 12.96%        | +10.5%      |

**Key Findings:**

- ✅ Hybrid Fixed beats baseline on all metrics
- ✅ Huge gains for cold-start users (+42%)
- ⚠️ Learnable fusion needs improvement
- ⚠️ Missing long-history user data

---

## 🎯 Next Steps (Priority Order)

### Week 1: Feb 17-23

**Day 1-2: Complete Missing Data**

- [ ] Re-run all experiments with long-history metrics
- [ ] Enable alpha value tracking
- [ ] Verify all 3 length bins present

**Day 3-4: Optimize Fixed Alpha**

- [ ] Grid search α ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
- [ ] Find optimal fixed fusion weight

**Day 5-6: Fix Learnable Fusion**

- [ ] Better initialization
- [ ] Add constraints (α ∈ [0.1, 0.9])
- [ ] Add L2 regularization

**Day 7: Improve Continuous Fusion**

- [ ] Try piecewise linear
- [ ] Try better sigmoid params

### Week 2: Feb 24-28

**Day 8-9: Advanced Improvements**

- [ ] More GNN layers (3-4 instead of 2)
- [ ] Hard negative mining
- [ ] Two-stage training

**Day 10: Analysis & Visualization**

- [ ] Run all analysis scripts
- [ ] Generate all plots
- [ ] Statistical significance tests

**Day 11-12: Teacher Report**

- [ ] Write method summary
- [ ] Create results tables
- [ ] Package code and results

**Buffer: Feb 29-Mar 1**

- Final checks and submission

---

## 📁 Project Structure

```
length-adaptive/
├── ACTION_PLAN.md              ← Strategy document
├── KAGGLE_EXPERIMENTS_TODO.md  ← Experiments to run
├── README.md                   ← Main project docs
├── experiments/
│   ├── README.md               ← Analysis tools guide
│   ├── quick_compare.py        ← Quick results comparison
│   ├── analyze_user_distribution.py
│   ├── statistical_tests.py
│   ├── create_visualizations.py
│   ├── run_all_analysis.sh     ← Run everything
│   └── setup_analysis_env.sh   ← Setup venv
├── src/
│   ├── eval/
│   │   ├── evaluator.py        ← Enhanced with alpha tracking
│   │   └── metrics.py          ← Supports 3 length bins
│   ├── models/
│   │   ├── fusion.py           ← Fusion mechanisms
│   │   ├── hybrid.py           ← Hybrid model
│   │   └── sasrec.py           ← Baseline
│   └── ...
└── results/                    ← Experiment results
    ├── sasrec_*/
    ├── hybrid_fixed_*/
    ├── hybrid_continuous_*/
    └── ...
```

---

## 🔍 Understanding the Approach

### The Problem

Traditional sequential recommenders (SASRec) treat all users equally:

- **Short-history users** (≤10 items): Not enough personalized data → poor performance
- **Long-history users** (>50 items): Rich personalized data → good performance

### Our Solution

**Length-Adaptive Fusion** combines:

1. **Global Collaborative (GNN)**
   - Item co-occurrence graph
   - "Users who liked A also liked B"
   - Helps cold-start users

2. **Personal Sequential (SASRec)**
   - Self-attention Transformer
   - Individual temporal patterns
   - Helps warm users

3. **Adaptive Weighting**

   ```
   h_i = α(u) × e_i + (1-α(u)) × g_i

   where α(u) depends on user history length:
   - Short users: α ≈ 0.3 (more GNN)
   - Medium users: α ≈ 0.5 (balanced)
   - Long users: α ≈ 0.7 (more SASRec)
   ```

---

## 📚 Key Documents

| Document                                                                                                                                                                                       | Purpose                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| [ACTION_PLAN.md](ACTION_PLAN.md)                                                                                                                                                               | Complete strategy & timeline |
| [KAGGLE_EXPERIMENTS_TODO.md](KAGGLE_EXPERIMENTS_TODO.md)                                                                                                                                       | Kaggle experiments queue     |
| [experiments/README.md](experiments/README.md)                                                                                                                                                 | Analysis tools usage         |
| [WORKFLOW.md](WORKFLOW.md)                                                                                                                                                                     | Git workflow & setup         |
| [Project - Hybrid GNN + Transformer Sequential Recommendation (MovieLens‑1M).md](Project%20-%20Hybrid%20GNN%20%2B%20Transformer%20Sequential%20Recommendation%20%28MovieLens%E2%80%911M%29.md) | Original project plan        |

---

## 🎓 For Teacher Submission

### What to Include

1. **Code Package**
   - Complete `src/` directory
   - Training scripts
   - Evaluation scripts
   - Requirements.txt

2. **Results**
   - All experiment JSON files
   - Comparison tables
   - Visualizations (PNG files)

3. **Documentation**
   - Method summary (1-2 pages)
   - Results summary (1 page)
   - Mathematical formulation

4. **Insights**
   - Performance by user length
   - Statistical significance
   - Ablation studies

### Key Message

> "Our length-adaptive hybrid GNN+SASRec model achieves substantial improvements for cold-start users (+42% HR@10) while maintaining overall performance gains (+3.7% HR@10) by intelligently balancing global collaborative signals and personalized sequential patterns based on user interaction history."

---

## 🛠️ Troubleshooting

### No venv directory

```bash
bash experiments/setup_analysis_env.sh
```

### matplotlib not found

```bash
source venv/bin/activate
pip install matplotlib seaborn
```

### No results found

Make sure you have experiment results in `results/` directory.
For now, use Kaggle for training, then download results.

### Permission denied

```bash
chmod +x experiments/*.sh
```

---

## 📞 Quick Commands

```bash
# Setup (first time)
bash experiments/setup_analysis_env.sh

# Activate environment
source venv/bin/activate

# Quick comparison
python experiments/quick_compare.py

# User distribution
python experiments/analyze_user_distribution.py

# All analyses
bash experiments/run_all_analysis.sh

# Visualizations
python experiments/create_visualizations.py
```

---

**Last Updated:** February 17, 2026  
**Deadline:** March 1, 2026 (12 days remaining)  
**Status:** 🟢 On track - Already beating baseline!
