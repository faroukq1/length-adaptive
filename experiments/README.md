# Analysis Scripts - Usage Guide

This directory contains scripts for analyzing experimental results.

## Setup

First, activate the virtual environment:

```bash
source venv/bin/activate
```

Install visualization dependencies (if not already installed):

```bash
pip install matplotlib seaborn
```

## Available Scripts

### 1. User Distribution Analysis

Analyzes how users are distributed across length bins (short/medium/long).

```bash
python experiments/analyze_user_distribution.py
```

**Output:**

- User counts and percentages per bin
- Statistics (mean, median, percentiles)
- Insights on dataset composition

---

### 2. Quick Results Comparison

Quick overview of all experiment results in a table format.

```bash
python experiments/quick_compare.py
```

**Output:**

- Overall performance comparison
- Short-history users comparison
- Medium-history users comparison
- Success criteria check

---

### 3. Detailed Results Analysis

Complete analysis with tables and detailed breakdowns.

```bash
python experiments/analyze_results.py
```

**Output:**

- Comparison tables
- Grouped performance metrics
- Model rankings

---

### 4. Statistical Significance Testing

Tests whether improvements are statistically significant.

```bash
python experiments/statistical_tests.py
```

**Output:**

- Hypothesis tests
- p-values
- Confidence intervals
- Significance indicators

---

### 5. Create Visualizations

Generates publication-quality plots.

```bash
python experiments/create_visualizations.py
```

**Output** (saved to `data/graphs/`):

- `performance_by_length.png` - Bar chart comparing models
- `alpha_function.png` - Alpha as function of sequence length
- `user_distribution.png` - Histogram of sequence lengths

---

## Run All Analyses

Run everything at once:

```bash
bash experiments/run_all_analysis.sh
```

This will:

1. Analyze user distribution
2. Compare all results
3. Run detailed analysis
4. Perform statistical tests
5. Create all visualizations

---

## For Kaggle Experiments

The training experiments should be run on Kaggle. Use the provided Kaggle notebook:

```bash
# See: kaggle_notebook.ipynb
```

After training on Kaggle:

1. Download results from Kaggle
2. Extract to `results/` directory
3. Run analysis scripts locally

---

## Troubleshooting

### matplotlib not found

```bash
pip install matplotlib seaborn
```

### Permission denied on .sh script

```bash
chmod +x experiments/run_all_analysis.sh
```

### No results found

Make sure you have experiment results in `results/` directory:

```bash
ls -la results/
```

Should show folders like:

- `sasrec_*/`
- `hybrid_fixed_*/`
- `hybrid_continuous_*/`
  etc.

---

## Analysis Checklist for Teacher Submission

- [ ] Run user distribution analysis
- [ ] Verify long-history metrics exist
- [ ] Generate all visualizations
- [ ] Run statistical tests
- [ ] Create comparison tables
- [ ] Document alpha values
- [ ] Package code and results

---

## Output Files

After running all scripts, you'll have:

```
data/graphs/
├── performance_by_length.png
├── alpha_function.png
└── user_distribution.png

results/
├── sasrec_*/
│   └── results.json
├── hybrid_fixed_*/
│   └── results.json
└── ...
```

---

## Quick Commands Reference

```bash
# Activate environment
source venv/bin/activate

# Run single analysis
python experiments/quick_compare.py

# Run all analyses
bash experiments/run_all_analysis.sh

# Create visualizations
python experiments/create_visualizations.py

# Check user distribution
python experiments/analyze_user_distribution.py

# Statistical tests
python experiments/statistical_tests.py
```

---

Last updated: February 17, 2026
