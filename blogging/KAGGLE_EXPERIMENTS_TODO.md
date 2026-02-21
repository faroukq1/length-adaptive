# Kaggle Experiments TODO - Beat SASRec Baseline

**Goal:** Improve hybrid model to consistently beat SASRec baseline on overall metrics.

**Current Status:**

- ✅ **Hybrid Fixed**: Best performer (HR@10: 9.99% vs 9.63% baseline)
- ⚠️ **Hybrid Learnable**: Underperforms (HR@10: 9.33%)
- ⚠️ **Hybrid Continuous**: Close but not best (HR@10: 9.61%)

---

## Priority 1: Complete Missing Data (CRITICAL)

### Check Long-History Users

**Issue:** Results only show "short" and "medium" groups, missing "long" group.

**Hypothesis:** Maybe no users have >50 interactions in test set?

**Action on Kaggle:**

```python
# In run_experiment.py, ensure evaluator uses:
metrics = evaluator.evaluate(
    test_loader,
    edge_index,
    edge_weight,
    compute_by_group=True,  # ← Must be True
    track_alpha=True        # ← Add alpha tracking
)

# Verify results contain 'long' key
print(f"Groups found: {list(metrics['grouped_metrics'].keys())}")
```

**Expected:** Should see "short", "medium", "long", "overall"

---

### Enable Alpha Tracking

**Action on Kaggle:**

```python
# In run_experiment.py, after evaluation:
if isinstance(eval_result, tuple):
    metrics, alpha_stats = eval_result

    # Save alpha stats
    with open(os.path.join(save_dir, 'alpha_stats.json'), 'w') as f:
        json.dump(alpha_stats, f, indent=2)
else:
    metrics = eval_result
```

---

## Priority 2: Optimize Fixed Alpha (HIGH IMPACT)

### Grid Search for Best α

**Hypothesis:** α=0.5 may not be optimal. Try range [0.3, 0.7].

**Experiments to Run:**

```python
alphas = [0.3, 0.4, 0.5, 0.6, 0.7]

for alpha in alphas:
    model = HybridSASRecGNN(
        ...,
        fusion_type='fixed',
        fixed_alpha=alpha
    )
    train(model, ...)
    save_results(f'hybrid_fixed_alpha{alpha}')
```

**Expected:** Find optimal α that maximizes overall HR@10.

**Time:** ~2 hours per α × 5 = 10 hours total

---

## Priority 3: Fix Learnable Fusion (HIGH IMPACT)

### Problem Analysis

Current learnable underperforms badly. Possible causes:

1. ❌ Bad initialization (random → unstable)
2. ❌ No constraints (α can go negative or >1)
3. ❌ Not enough regularization

### Improved Implementation

**Update `src/models/fusion.py`:**

```python
class LearnableFusion(nn.Module):
    """Improved learnable fusion with better initialization and constraints"""

    def __init__(self, L_short=10, L_long=50,
                 init_short=0.4, init_mid=0.5, init_long=0.6):
        super().__init__()
        self.L_short = L_short
        self.L_long = L_long

        # Initialize near optimal fixed values
        self.alpha_short = nn.Parameter(torch.tensor(init_short))
        self.alpha_mid = nn.Parameter(torch.tensor(init_mid))
        self.alpha_long = nn.Parameter(torch.tensor(init_long))

    def compute_alpha(self, lengths):
        batch_size = lengths.size(0)
        device = lengths.device
        alphas = torch.zeros(batch_size, 1, device=device)

        short_mask = lengths <= self.L_short
        long_mask = lengths > self.L_long
        mid_mask = ~(short_mask | long_mask)

        # Clamp to [0.1, 0.9] to enforce constraints
        alpha_s = torch.clamp(self.alpha_short, 0.1, 0.9)
        alpha_m = torch.clamp(self.alpha_mid, 0.1, 0.9)
        alpha_l = torch.clamp(self.alpha_long, 0.1, 0.9)

        alphas[short_mask] = alpha_s
        alphas[mid_mask] = alpha_m
        alphas[long_mask] = alpha_l

        return alphas
```

**Add L2 Regularization:**

```python
# In trainer.py, add to loss:
if hasattr(model, 'fusion') and hasattr(model.fusion, 'alpha_short'):
    alpha_reg = (model.fusion.alpha_short**2 +
                 model.fusion.alpha_mid**2 +
                 model.fusion.alpha_long**2)
    loss = loss + 0.01 * alpha_reg  # Small L2 penalty
```

**Time:** ~2 hours to test

---

## Priority 4: Improve Continuous Fusion (MEDIUM IMPACT)

### Current Problem

Sigmoid-based continuous function too smooth, doesn't adapt well.

### Better Alternatives

#### Option A: Piecewise Linear

```python
class PiecewiseLinearFusion(nn.Module):
    """Linear interpolation between learned bin values"""

    def __init__(self):
        super().__init__()
        # Learn alpha at key points
        self.alpha_5 = nn.Parameter(torch.tensor(0.3))
        self.alpha_15 = nn.Parameter(torch.tensor(0.4))
        self.alpha_30 = nn.Parameter(torch.tensor(0.5))
        self.alpha_60 = nn.Parameter(torch.tensor(0.6))

    def compute_alpha(self, lengths):
        # Linear interpolation between points
        alphas = torch.zeros_like(lengths, dtype=torch.float32)

        for i, L in enumerate(lengths):
            if L <= 5:
                alphas[i] = torch.clamp(self.alpha_5, 0.1, 0.9)
            elif L <= 15:
                # Interpolate between 5 and 15
                t = (L - 5) / 10.0
                alphas[i] = (1-t) * self.alpha_5 + t * self.alpha_15
            elif L <= 30:
                t = (L - 15) / 15.0
                alphas[i] = (1-t) * self.alpha_15 + t * self.alpha_30
            else:
                t = min((L - 30) / 30.0, 1.0)
                alphas[i] = (1-t) * self.alpha_30 + t * self.alpha_60

        return alphas.unsqueeze(1)
```

#### Option B: Learned Sigmoid with Better Init

```python
class ImprovedContinuousFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize to approximate discrete bins
        self.w = nn.Parameter(torch.tensor(-0.1))  # Negative for increasing
        self.b = nn.Parameter(torch.tensor(1.5))
        self.scale = nn.Parameter(torch.tensor(0.4))  # Range control
        self.offset = nn.Parameter(torch.tensor(0.5))  # Midpoint

    def compute_alpha(self, lengths):
        # Constrained sigmoid
        x = self.w * torch.log(lengths.float() + 1) + self.b
        alpha = torch.sigmoid(x) * self.scale + self.offset
        return alpha.unsqueeze(1).clamp(0.1, 0.9)
```

**Time:** ~2 hours each = 4 hours total

---

## Priority 5: Enhanced GNN (ADVANCED)

### Current: Simple LightGCN

**Improvements to Try:**

#### A. Add More GNN Layers

```python
model = HybridSASRecGNN(
    ...,
    gnn_layers=3,  # Currently 2, try 3-4
    ...
)
```

#### B. Attention-Based GNN

```python
# Replace LightGCN with GAT (Graph Attention Network)
from torch_geometric.nn import GATConv

class AttentionGNN(nn.Module):
    def __init__(self, num_items, d_model, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, d_model)
        self.convs = nn.ModuleList([
            GATConv(d_model, d_model, heads=4, concat=False)
            for _ in range(n_layers)
        ])

    def forward(self, edge_index, edge_weight=None):
        x = self.embedding.weight
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x
```

**Time:** ~4 hours to test

---

## Priority 6: Better Training Strategy (MEDIUM IMPACT)

### A. More Negative Samples

**Current:** 1 negative per positive

**Try:** 4 negatives per positive

```python
# In trainer.py
def sample_negatives(batch, num_negatives=4):
    ...
```

**Expected:** Better gradient signal, especially for hybrid model.

---

### B. Hard Negative Mining

**Strategy:** Sample popular items user hasn't seen, not random.

```python
def hard_negative_sampling(user_history, item_popularity, num_negatives=4):
    # Get popular items not in user history
    candidates = [item for item in item_popularity[:100]
                  if item not in user_history]
    return random.sample(candidates, num_negatives)
```

---

### C. Two-Stage Training

**Stage 1:** Pre-train GNN alone

```python
# Train LightGCN on BPR loss for 10 epochs
# Save GNN embeddings
```

**Stage 2:** Train hybrid with frozen GNN

```python
# Load pre-trained GNN
# Freeze GNN parameters
# Train fusion + SASRec
```

**Expected:** Better GNN embeddings → better fusion.

**Time:** ~6 hours total

---

## Experiment Priority Queue

### Round 1: Must-Do (Est. 12-14 hours)

1. ✅ **Re-run all with long-history + alpha tracking** (10h)
   - All 5 existing models
   - Verify long-history metrics exist
2. ✅ **Fixed alpha grid search** (10h)
   - Test α ∈ {0.3, 0.4, 0.5, 0.6, 0.7}

### Round 2: High-Priority Fixes (Est. 8-10 hours)

3. ✅ **Improved learnable fusion** (2h)
   - Better init + constraints
4. ✅ **Piecewise linear fusion** (2h)
   - Replace continuous with piecewise

5. ✅ **Improved continuous fusion** (2h)
   - Better sigmoid parameterization

6. ✅ **More GNN layers** (2h)
   - Try 3-4 layers instead of 2

### Round 3: Advanced (If Time Permits - Est. 10-12 hours)

7. ⭐ **Hard negative mining** (4h)
8. ⭐ **Two-stage training** (6h)
9. ⭐ **GAT-based GNN** (4h)

---

## Expected Improvements

### Conservative Estimate

- **Fixed α optimized:** +1-2% over current best
- **Improved learnable:** Match or beat fixed
- **Better continuous:** Match fixed

**Target:** HR@10 ≥ 10.5% (vs 9.63% baseline = +9% improvement)

### Optimistic Estimate

- **All improvements combined:** +3-5% over baseline
- **Strong short-user gains:** +50-60%

**Target:** HR@10 ≥ 11% (vs 9.63% baseline = +14% improvement)

---

## Testing Strategy

### Quick Test (Before Full Run)

```python
# Use small subset to verify code works
train_subset = train_sequences[:1000]
val_subset = val_sequences[:200]

# Train for 5 epochs
# Check:
# - No errors
# - Loss decreases
# - Alpha values in expected range
```

### Full Run Checklist

- [x] Data loaded correctly
- [x] Model initialized
- [x] Training converges
- [x] Validation metrics computed
- [x] Test metrics saved with:
  - [x] 'short', 'medium', 'long', 'overall' groups
  - [x] Alpha statistics (for hybrid models)
- [x] Best model saved

---

## Code Changes Needed

### 1. Update `run_experiment.py`

```python
# Add alpha tracking
val_result = evaluator.evaluate(
    val_loader, edge_index, edge_weight,
    compute_by_group=True,
    track_alpha=True if 'hybrid' in args.model else False
)

# Handle tuple return
if isinstance(val_result, tuple):
    val_metrics, alpha_stats = val_result
    # Save alpha stats
else:
    val_metrics = val_result
```

### 2. Update `fusion.py`

- Add improved learnable fusion class
- Add piecewise linear fusion class
- Add improved continuous fusion class

### 3. Update `hybrid.py`

```python
# Support new fusion types
FUSION_TYPES = {
    'fixed': FixedFusion,
    'discrete': DiscreteFusion,
    'learnable': ImprovedLearnableFusion,  # New
    'continuous': ImprovedContinuousFusion,  # New
    'piecewise': PiecewiseLinearFusion,  # New
}
```

---

## Results to Save

For each experiment, save:

```json
{
  "test_metrics": {
    "HR@5": ..., "HR@10": ..., "HR@20": ...,
    "NDCG@5": ..., "NDCG@10": ..., "NDCG@20": ...,
    "MRR@5": ..., "MRR@10": ..., "MRR@20": ...
  },
  "grouped_metrics": {
    "short": { "HR@10": ..., "count": ... },
    "medium": { "HR@10": ..., "count": ... },
    "long": { "HR@10": ..., "count": ... },
    "overall": { "HR@10": ..., "count": ... }
  },
  "alpha_stats": {  // Only for hybrid models
    "short": { "mean": ..., "std": ..., "count": ... },
    "medium": { "mean": ..., "std": ..., "count": ... },
    "long": { "mean": ..., "std": ..., "count": ... }
  },
  "best_epoch": ...,
  "best_val_metric": ...
}
```

---

## Success Criteria

### Minimum (Must Achieve)

- ✅ Overall HR@10 > 10.0% (+3.8% vs baseline)
- ✅ Short HR@10 > 18% (+53% vs baseline)
- ✅ Long-history data collected
- ✅ Statistical significance (p < 0.05)

### Target (Good Result)

- ⭐ Overall HR@10 > 10.5% (+9% vs baseline)
- ⭐ Short HR@10 > 20% (+70% vs baseline)
- ⭐ All fusion types tested
- ⭐ Ablation study complete

### Stretch (Excellent Result)

- 🌟 Overall HR@10 > 11% (+14% vs baseline)
- 🌟 Short HR@10 > 22% (+87% vs baseline)
- 🌟 Publication-quality results
- 🌟 Novel fusion mechanism validated

---

## Timeline

**Feb 17-20:** Round 1 experiments  
**Feb 21-23:** Round 2 experiments  
**Feb 24-26:** Round 3 experiments (if time)  
**Feb 27:** Analysis and visualization  
**Feb 28:** Write teacher report  
**Mar 1:** Final submission

---

Last updated: February 17, 2026
