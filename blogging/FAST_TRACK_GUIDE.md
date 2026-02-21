# ⚡ FAST TRACK - Kaggle Notebook Run Guide

**Total Time: ~13 minutes (instead of 45+ minutes)**

---

## 🚀 Cells to Run (In Order)

### 1️⃣ Cell 1 - Title & Instructions

**Type:** Markdown  
**Time:** 0 sec  
**Action:** Just read it

---

### 2️⃣ Cell 2 - Clone Repository

**Type:** Markdown  
**Time:** 0 sec  
**Action:** Just read it

---

### 3️⃣ Cell 3 - Clone Repo Command

**Type:** Python  
**Time:** ~2 minutes  
**Action:** ✅ **RUN THIS**

```python
!git clone https://github.com/faroukq1/length-adaptive.git
%cd length-adaptive
...
```

---

### 4️⃣ Cell 4 - Install Dependencies Header

**Type:** Markdown  
**Time:** 0 sec  
**Action:** Just read it

---

### 5️⃣ Cell 5 - Install Packages

**Type:** Python  
**Time:** ~1 minute  
**Action:** ✅ **RUN THIS**

```python
!pip install -q torch-geometric tqdm scikit-learn pandas matplotlib
```

---

### 6️⃣ Cell 6 - GPU Check Header

**Type:** Markdown  
**Time:** 0 sec  
**Action:** Just read it

---

### 7️⃣ Cell 7 - Check GPU

**Type:** Python  
**Time:** ~5 seconds  
**Action:** ✅ **RUN THIS** (quick, useful to verify GPU)

```python
!python check_gpu.py
```

---

### 8️⃣ Cell 8 - Quick Test Header

**Type:** Markdown  
**Time:** 0 sec  
**Action:** Just read it (says to skip)

---

### 9️⃣ Cell 9 - Quick Test Command

**Type:** Python  
**Time:** Would be 2 min  
**Action:** ❌ **SKIP THIS** (already set to skip)

```python
# SKIP: Quick test (saves 2 minutes)
print("⚡ Skipped...")
```

---

### 🔟 Cell 10 - Train Hybrid Header

**Type:** Markdown  
**Time:** 0 sec  
**Action:** Just read it

---

### 1️⃣1️⃣ Cell 11 - Train Hybrid Command ⭐ CRITICAL

**Type:** Python  
**Time:** ~10 minutes  
**Action:** ✅ **RUN THIS** - This is the main experiment!

```python
!python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs 50 ...
```

**Note:** This is the most important cell. Wait for it to complete.

---

### 1️⃣2️⃣ Cell 12 - SASRec Baseline Header

**Type:** Markdown  
**Time:** 0 sec  
**Action:** Just read it (says to skip)

---

### 1️⃣3️⃣ Cell 13 - SASRec Training

**Type:** Python  
**Time:** Would be 8 min  
**Action:** ❌ **SKIP THIS** (you already have baseline!)

```python
# OPTION 1: Skip SASRec training
print("💡 Skipping SASRec...")
```

---

### 1️⃣4️⃣ Cell 14+ - Quick Results View ⭐ NEW

**Type:** Python  
**Time:** ~1 second  
**Action:** ✅ **RUN THIS** - See if you beat baseline!

```python
# Quick performance check
...shows HR@10 comparison...
```

---

### Skip to Cell ~22 - Download Results

**Type:** Python  
**Time:** ~30 seconds  
**Action:** ✅ **RUN THIS** - Download results.zip

```python
!zip -r results.zip results/
```

---

## 📋 Quick Checklist

```
□ Cell 3  - Clone repo (2 min)
□ Cell 5  - Install deps (1 min)
□ Cell 7  - GPU check (5 sec)
✗ Cell 9  - SKIP quick test
□ Cell 11 - Train hybrid (10 min) ⭐ CRITICAL
✗ Cell 13 - SKIP SASRec
□ Cell 14+ - Quick results (1 sec)
□ Cell ~22 - Download (30 sec)
────────────────────────────────
Total: ~13 minutes
```

---

## 🎯 Expected Output

After running Cell 11 (hybrid training), you should see:

```
======================================================================
🚀 Training Hybrid Discrete Model
======================================================================

[1/5] Loading preprocessed data...
  Users: 6,040
  Items: 3,706

[2/5] Loading co-occurrence graph...
  Edges: 151,874

[3/5] Creating dataloaders...

[4/5] Creating model...
  Model: HybridSASRecGNN
  Fusion: discrete

[5/5] Initializing trainer...

============================================================
TRAINING
============================================================
Epoch 1/50: 100%|███████| train_loss=2.3456 val_NDCG@10=0.0234
Epoch 2/50: 100%|███████| train_loss=1.9876 val_NDCG@10=0.0345
...
Epoch 28/50: 100%|██████| train_loss=0.8765 val_NDCG@10=0.0471
⭐ New best! Saving checkpoint...

Early stopping triggered at epoch 28

============================================================
TESTING
============================================================
HR@10: 0.0996
NDCG@10: 0.0471
MRR@10: 0.0286

✅ Experiment complete! Results saved to: results/hybrid_discrete_20260217_123456
```

---

## ⚡ After Training

1. **Quick Check** - Run Cell 14+ to see improvement over baseline
2. **Download** - Run Cell ~22 to download results.zip
3. **Local Analysis** - Extract and run full analysis scripts locally

---

## 🔧 If Something Goes Wrong

**GPU not available?**

- Settings → Accelerator → GPU T4
- Restart notebook

**Clone failed?**

- Settings → Internet → On
- Check GitHub repo is public

**Training too slow?**

- Verify GPU is enabled (should say "cuda" in Cell 7)
- If on CPU, expect ~40 minutes instead of 10

**Out of memory?**

- Reduce batch size: `--batch_size 128` instead of 256
- Restart notebook and try again

---

## 💾 What You'll Download

`results.zip` contains:

```
results/
└── hybrid_discrete_YYYYMMDD_HHMMSS/
    ├── best_model.pt        # Trained model
    ├── results.json         # Test metrics ⭐
    ├── history.json         # Training progress
    └── config.json          # Hyperparameters
```

Extract locally and run:

```bash
cd /path/to/project
bash scripts/merge_kaggle_results.sh ~/Downloads/results.zip
python experiments/quick_compare.py
```

---

**Ready to go!** Just run cells 3, 5, 7, 11, and the download cell. That's it! 🚀
