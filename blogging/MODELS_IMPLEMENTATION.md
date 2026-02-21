# Models Implementation Summary

## ✅ Implemented Models (8 Total)

### Baseline Models

#### 1. SASRec (Self-Attentive Sequential Recommendation)

- **File**: `src/models/sasrec.py`
- **Architecture**: Transformer with causal masking
- **Key Features**: Multi-head self-attention, position embeddings
- **Reference**: ICDM 2018

#### 2. BERT4Rec (Bidirectional Transformer)

- **File**: `src/models/bert4rec.py` ✨ NEW
- **Architecture**: Bidirectional Transformer (no causal masking)
- **Key Features**: GELU activation, bidirectional attention
- **Reference**: CIKM 2019

#### 3. GRU4Rec (RNN-based Sequential)

- **File**: `src/models/gru4rec.py` ✨ NEW
- **Architecture**: Multi-layer GRU with packed sequences
- **Key Features**: Efficient RNN processing, variable-length handling
- **Reference**: ICLR 2016

#### 4. LightGCN (Graph Neural Network for Sequential)

- **File**: `src/models/lightgcn_seq.py` ✨ NEW
- **Architecture**: LightGCN + Attention-based sequence pooling
- **Key Features**: Graph-enhanced embeddings, precomputed GNN layers
- **Reference**: SIGIR 2020 (adapted for sequential recommendation)

### Hybrid Models (Ours)

#### 5. Hybrid Fixed (α=0.5)

- **File**: `src/models/hybrid.py`
- **Architecture**: SASRec + LightGCN with fixed fusion weight
- **Fusion**: `output = 0.5 * sequential + 0.5 * graph`

#### 6. Hybrid Discrete (Bin-based)

- **File**: `src/models/hybrid.py`
- **Architecture**: SASRec + LightGCN with bin-based fusion
- **Fusion**: Different weights for short/medium/long history users

#### 7. Hybrid Learnable (MLP-based)

- **File**: `src/models/hybrid.py`
- **Architecture**: SASRec + LightGCN with learned fusion weights
- **Fusion**: MLP predicts fusion weight from sequence length

#### 8. Hybrid Continuous (Sigmoid-based)

- **File**: `src/models/hybrid.py`
- **Architecture**: SASRec + LightGCN with smooth fusion
- **Fusion**: Sigmoid function for smooth transition

---

## 🔧 Updated Files

### Core Training Framework

1. **`experiments/run_experiment.py`**
   - Added support for: `bert4rec`, `gru4rec`, `lightgcn`
   - Updated model choices in argument parser
   - Added imports for new models

2. **`src/train/trainer.py`**
   - Added LightGCN-specific handling
   - Precomputes graph embeddings for LightGCN (efficiency)
   - Updated forward pass logic for each model type
   - Fixed evaluation to handle graph embeddings

3. **`src/eval/evaluator.py`**
   - Added LightGCN detection
   - Passes precomputed graph embeddings during evaluation
   - Handles different model architectures transparently

### Experiment Scripts

4. **`scripts/run_paper_experiments.sh`**
   - Expanded from 5 to 8 models
   - Added experiments for BERT4Rec, GRU4Rec, LightGCN
   - Updated time estimates (8-10 hours vs 3-4 hours)
   - Added GNN_LAYERS parameter
   - Updated all hybrid models to include gnn_layers parameter

### Kaggle Notebook

5. **`kaggle_paper.ipynb`**
   - Updated to show 8 models (4 baselines + 4 hybrid)
   - Enhanced comparison sections
   - Added baseline vs hybrid categorization
   - Updated time estimates and descriptions
   - Improved results display with model types
   - Added comprehensive comparison analysis

---

## 📊 Fair Comparison Configuration

All models use **identical hyperparameters** for fair comparison:

```bash
EPOCHS=200
PATIENCE=20
BATCH_SIZE=256
LR=0.001
MAX_LEN=50
D_MODEL=64
N_HEADS=2
N_BLOCKS=2
GNN_LAYERS=2
```

### Model-Specific Notes:

- **SASRec, BERT4Rec**: Use n_heads and n_blocks for Transformer
- **GRU4Rec**: Uses n_blocks as n_layers for GRU
- **LightGCN**: Uses gnn_layers for graph convolutions
- **Hybrid Models**: Use both Transformer and GNN parameters

---

## 🚀 How to Run

### Local Training

```bash
# Single model
python experiments/run_experiment.py --model bert4rec --epochs 50

# All models (paper settings)
bash scripts/run_paper_experiments.sh
```

### Kaggle Training

1. Open `kaggle_paper.ipynb` in Kaggle
2. Enable GPU T4 accelerator
3. Enable Internet access
4. Run all cells sequentially
5. Download `results_paper.zip` when complete

---

## 📈 Expected Results

### Baseline Model Comparison

- **BERT4Rec**: May perform better due to bidirectional context
- **SASRec**: Strong unidirectional baseline
- **GRU4Rec**: Older but efficient RNN baseline
- **LightGCN**: Pure graph-based approach

### Hybrid Model Performance

- **Goal**: Outperform best baseline by combining sequential + graph
- **Hypothesis**: Fusion helps especially for short-history users
- **Trade-off**: May sacrifice warm-user performance for cold-start gains

---

## 🔬 Analysis Features

### Overall Metrics

- HR@5, HR@10, HR@20
- NDCG@5, NDCG@10, NDCG@20
- MRR@10

### User Group Analysis

- **Short**: ≤10 interactions (cold-start users)
- **Medium**: 11-50 interactions
- **Long**: >50 interactions (warm users)

### Visualization

- Training loss curves
- Validation NDCG@10 curves
- Early stopping markers
- Model comparison tables

---

## ✅ Testing

Quick model test (without training):

```bash
python test_models.py
```

This will:

- Instantiate all 8 models
- Run forward pass with dummy data
- Verify output shapes
- Count parameters

---

## 📝 Model Parameters Count

Approximate parameter counts (num_items=3706, d_model=64):

- **SASRec**: ~250K parameters
- **BERT4Rec**: ~250K parameters
- **GRU4Rec**: ~170K parameters (fewer layers)
- **LightGCN**: ~240K parameters
- **Hybrid Models**: ~320K parameters (Sequential + GNN)

---

## 🎯 Next Steps

1. **Run Experiments**: Execute `run_paper_experiments.sh`
2. **Analyze Results**: Check which baseline is strongest
3. **Compare Hybrid**: Evaluate if fusion helps overall or specific groups
4. **Document Findings**: Prepare presentation/report
5. **Consider Extensions**: Meta-learning, personalized graphs, etc.

---

## 📚 References

- **SASRec**: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018
- **BERT4Rec**: Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer", CIKM 2019
- **GRU4Rec**: Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks", ICLR 2016
- **LightGCN**: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020
