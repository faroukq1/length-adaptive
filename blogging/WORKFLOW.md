# Workflow: From Quick Test to Paper-Level Results

This guide shows you how to progress from quick testing to achieving paper-level performance.

---

## 🎯 Overview

| Phase                        | Purpose             | Time (GPU) | Expected NDCG@10 |
| ---------------------------- | ------------------- | ---------- | ---------------- |
| **1. Pipeline Test**         | Verify code works   | 2 min      | ~0.02-0.03       |
| **2. Quick Baseline**        | Fast iteration      | 10 min     | ~0.10-0.15       |
| **3. Paper Training**        | Match paper results | 30-40 min  | ~0.18-0.22       |
| **4. Hyperparameter Tuning** | Beat paper results  | 3-5 hours  | ~0.20-0.25       |

---

## Phase 1: Pipeline Test ✅ (YOU ARE HERE)

**Purpose:** Verify everything works end-to-end

```bash
# Test with 2 epochs (~2 min)
python test_training.py
```

**Expected Output:**

- SASRec: NDCG@10 ~0.015-0.020
- Hybrid: NDCG@10 ~0.023-0.030

**What this tells you:** Code runs without errors ✅

---

## Phase 2: Quick Baseline (10 minutes)

**Purpose:** Get quick results for debugging/iteration

```bash
# Single model - 50 epochs with early stopping
python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs 50 \
    --patience 10
```

**Expected Output:** NDCG@10 ~0.10-0.15

**What this tells you:**

- ✅ Training works properly
- ✅ Model is learning
- ⚠️ Results lower than paper (expected!)

**Why lower?**

- Early stopping at ~20-25 epochs (not fully converged)
- Small epoch budget prevents full optimization

---

## Phase 3: Paper-Level Training 📈 (30-40 minutes)

**Purpose:** Match paper results

### Option A: Single Model

```bash
python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs 200 \
    --patience 20 \
    --batch_size 256 \
    --lr 0.001
```

### Option B: All Models (Recommended)

```bash
chmod +x scripts/run_paper_experiments.sh
bash scripts/run_paper_experiments.sh
```

**Expected Output:** NDCG@10 ~0.18-0.22

**What happens:**

- Training runs for up to 200 epochs
- Early stopping patience=20 (waits 20 epochs for improvement)
- Usually converges at epoch 30-50
- Saves best model automatically

**Expected Results (on MovieLens-1M):**

| Model               | NDCG@10   | HR@10     | MRR@10    |
| ------------------- | --------- | --------- | --------- |
| SASRec              | 0.183     | 0.370     | 0.165     |
| Hybrid (Fixed)      | 0.195     | 0.387     | 0.172     |
| Hybrid (Discrete)   | **0.216** | **0.412** | **0.189** |
| Hybrid (Learnable)  | 0.210     | 0.405     | 0.186     |
| Hybrid (Continuous) | 0.213     | 0.409     | 0.188     |

---

## Phase 4: Hyperparameter Tuning 🏆 (Optional)

**Purpose:** Beat paper results or optimize for your dataset

### 4.1: Tune Learning Rate

```bash
# Try lower LR
python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs 200 \
    --lr 0.0005 \
    --patience 25

# Try higher LR
python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs 200 \
    --lr 0.002 \
    --patience 20
```

### 4.2: Tune Model Size

```bash
# Larger model (more capacity)
python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs 200 \
    --d_model 128 \
    --n_heads 4 \
    --n_blocks 3 \
    --lr 0.001

# Smaller model (faster, less overfitting)
python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs 200 \
    --d_model 32 \
    --n_heads 2 \
    --n_blocks 1 \
    --lr 0.001
```

### 4.3: Tune Batch Size

```bash
# Larger batch (more stable gradients)
python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs 200 \
    --batch_size 512 \
    --lr 0.002

# Smaller batch (more updates per epoch)
python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.0005
```

### 4.4: Tune Fusion Thresholds (Hybrid models only)

```bash
python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs 200 \
    --L_short 5 \
    --L_long 30
```

---

## Analyzing Results

```bash
# Generate comparison tables
python experiments/analyze_results.py --save_csv

# View results
cat results/overall_comparison.csv
```

---

## Quick Reference: Commands

```bash
# 1. Quick test (2 min)
python test_training.py

# 2. Quick baseline (10 min)
python experiments/run_experiment.py --model hybrid_discrete --epochs 50

# 3. Paper-level single model (30 min)
python experiments/run_experiment.py --model hybrid_discrete --epochs 200 --patience 20

# 4. Paper-level all models (3-4 hours)
bash scripts/run_paper_experiments.sh

# 5. Analyze
python experiments/analyze_results.py --save_csv
```

---

## Expected Performance Progression

| Stage          | Epochs       | Train Time | NDCG@10 | Notes               |
| -------------- | ------------ | ---------- | ------- | ------------------- |
| Pipeline Test  | 2            | 2 min      | 0.026   | Just testing        |
| Quick Baseline | 50 (→20-25)  | 10 min     | 0.12    | Fast iteration      |
| Paper Training | 200 (→30-50) | 30 min     | 0.21    | Publication quality |
| Tuned          | 200 (→40-60) | varies     | 0.23    | Optimized           |

---

## Troubleshooting

### Results still lower than expected?

1. **Check early stopping:**

   ```bash
   # Increase patience
   --patience 30
   ```

2. **Try more epochs:**

   ```bash
   --epochs 300
   ```

3. **Check for overfitting:**
   - Look at training vs validation curves
   - If val metric decreases while train improves → overfitting
   - Solution: Add dropout or reduce model size

4. **Verify data:**
   ```bash
   python -c "import pickle; data=pickle.load(open('data/ml-1m/processed/sequences.pkl','rb')); print('Users:', data['config']['num_users'])"
   ```
   Should show: Users: 6,034

### Training too slow?

1. **Enable GPU** (if not already)
2. **Reduce batch size** if OOM
3. **Reduce model size** for faster iteration

---

## Summary

**To get paper-level results:**

1. ✅ Verify pipeline works (2 min) → You did this!
2. 🎯 Run paper-level training:
   ```bash
   bash scripts/run_paper_experiments.sh
   ```
3. 📊 Analyze results:
   ```bash
   python experiments/analyze_results.py
   ```
4. 🏆 (Optional) Tune hyperparameters to beat paper

**Expected final NDCG@10:** ~0.21-0.22 for Hybrid (Discrete)

Good luck! 🚀
