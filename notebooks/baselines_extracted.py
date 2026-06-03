# Notebook: notebooks/baselines.ipynb
# Cells: 26

## MARKDOWN CELL 0
# Paper Results Collection: Baseline Models Only

**Publication-Ready Results for MovieLens-1M**

---

## 📋 Notebook Purpose

This notebook collects **baseline model results** for comparison:
- **5 Baseline Models:** BERT4Rec, SASRec, GRU4Rec, Caser, LightGCN

**What this notebook generates:**
1. ✅ Comprehensive performance comparison tables (5 baseline models)
2. ✅ Training dynamics analysis (convergence, epochs)
3. ✅ Performance by sequence length (short vs medium)
4. ✅ Computational cost analysis (time, memory, parameters)
5. ✅ Hyperparameter details
6. ✅ Publication-ready figures and tables
7. ✅ LaTeX-formatted tables for paper

**Models Evaluated (5 total):**

**Baselines:**
1. BERT4Rec (bidirectional transformer)
2. SASRec (unidirectional transformer)
3. GRU4Rec (RNN-based)
4. Caser (CNN-based)
5. LightGCN (pure GNN)

**Time: ~7-10 hours on GPU T4**

---

## 🎯 Quick Start

1. **Enable GPU T4** (Runtime → Change runtime type)
2. **Enable Internet**
3. **Run all cells** sequentially
4. **Download paper_results.zip** at the end

## MARKDOWN CELL 1
## Step 1: Setup Environment

## CODE CELL 2
# Clone repository
!git clone https://github.com/faroukq1/length-adaptive.git
%cd length-adaptive

# Verify repository structure
!ls -lh scripts/

print("\n✅ Repository cloned successfully!")

## MARKDOWN CELL 3
## Step 2: Install Dependencies

## CODE CELL 4
# Install required packages
!pip install -q torch-geometric tqdm scikit-learn pandas matplotlib seaborn scipy

# Verify GPU availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n✅ All dependencies installed!")

## MARKDOWN CELL 5
## Step 3: Prepare Data

## CODE CELL 6
import os

# Check if preprocessed data exists
data_file = 'data/ml-1m/processed/sequences.pkl'
graph_file = 'data/graphs/cooccurrence_graph.pkl'

print("="*70)
print("🔍 Checking Data Files")
print("="*70)

if os.path.exists(data_file):
    print(f"✅ Sequential data found: {data_file}")
    print(f"   Size: {os.path.getsize(data_file) / 1024 / 1024:.2f} MB")
else:
    print(f"❌ Sequential data NOT found: {data_file}")

if os.path.exists(graph_file):
    print(f"✅ Graph data found: {graph_file}")
    print(f"   Size: {os.path.getsize(graph_file) / 1024 / 1024:.2f} MB")
else:
    print(f"❌ Graph data NOT found: {graph_file}")

print("="*70)

# If data is missing, download and preprocess
if not os.path.exists(data_file) or not os.path.exists(graph_file):
    print("\n🔧 Downloading and preprocessing MovieLens-1M...")
    print("This will take 2-3 minutes.\n")
    
    # Download MovieLens-1M
    raw_file = 'data/ml-1m/raw/ml-1m/ratings.dat'
    if not os.path.exists(raw_file):
        print("📥 Downloading MovieLens-1M dataset...")
        !mkdir -p data/ml-1m/raw
        !wget -q http://files.grouplens.org/datasets/movielens/ml-1m.zip
        !unzip -q ml-1m.zip
        !mv ml-1m data/ml-1m/raw/
        !rm -f ml-1m.zip
        print("✅ Download complete!\n")
    
    # Preprocess sequences
    print("🔄 Preprocessing sequential data...")
    !python -m src.data.preprocess
    
    # Build graph
    print("\n🔄 Building co-occurrence graph...")
    !python -m src.data.graph_builder
    
    print("\n✅ Data preprocessing complete!")
else:
    print("\n✅ All data files ready!")

print("="*70)

## MARKDOWN CELL 7
## Step 4: Run Baseline Experiments

**Training 5 baseline models with 200 epochs, early stopping (patience=20)**

This is the longest step (~7-10 hours total with GPU T4)

## CODE CELL 8
import time
import os

print("="*80)
print("🎓 TRAINING BASELINE MODELS")
print("="*80)
print("")
print("Training 5 baseline models:")
print("")
print("BASELINES:")
print("  1. BERT4Rec (bidirectional transformer)")
print("  2. SASRec (unidirectional transformer)")
print("  3. GRU4Rec (RNN-based)")
print("  4. Caser (CNN-based)")
print("  5. LightGCN (pure GNN)")
print("")
print("Configuration:")
print("  - Max epochs: 200")
print("  - Early stopping patience: 20")
print("  - Batch size: 256")
print("  - Learning rate: 0.001")
print("  - d_model: 64, n_heads: 2, n_blocks: 2, gnn_layers: 2")
print("")
print("⏱️  Estimated time: 7-10 hours with GPU T4")
print("="*80)

# Record start time
start_time = time.time()

# Run baseline experiments only
!bash scripts/run_baselines.sh

# Record end time
total_time = time.time() - start_time
print("\n" + "="*80)
print(f"✅ All baseline experiments complete!")
print(f"⏱️  Total time: {total_time/3600:.2f} hours")
print("="*80)

## MARKDOWN CELL 9
## Step 5: Collect Results - Performance Metrics

Load all experimental results and create comprehensive comparison tables

## CODE CELL 10
!zip -r /kaggle/working/.virtual_documents/results.zip /kaggle/working/length-adaptive/results

## CODE CELL 11
import json
import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("📊 COLLECTING EXPERIMENTAL RESULTS")
print("="*80)

results_dir = Path('')

# Baseline models only
all_models = [
    'bert4rec',
    'sasrec',
    'gru4rec',
    'caser',
    'lightgcn'
]

# Collect all results
all_results = []
per_user_metrics = {}

for result_folder in sorted(results_dir.glob('*')):
    if result_folder.is_dir():
        results_file = result_folder / 'results.json'
        config_file = result_folder / 'config.json'
        history_file = result_folder / 'history.json'
        
        if results_file.exists() and config_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            with open(config_file) as f:
                config = json.load(f)
            
            model_name = config['model']
            
            # Only include baseline models with 200 epochs
            if model_name in all_models and config.get('epochs') == 200:
                
                # Load training history if available
                best_val_history = []
                if history_file.exists():
                    with open(history_file) as f:
                        history = json.load(f)
                        best_val_history = history.get('val_metrics', [])
                
                result_data = {
                    'Model': model_name,
                    'Model_Type': 'Baseline',
                    'Folder': result_folder.name,
                    
                    # Overall metrics
                    'HR@5': results['test_metrics']['HR@5'],
                    'HR@10': results['test_metrics']['HR@10'],
                    'HR@20': results['test_metrics']['HR@20'],
                    'NDCG@5': results['test_metrics']['NDCG@5'],
                    'NDCG@10': results['test_metrics']['NDCG@10'],
                    'NDCG@20': results['test_metrics']['NDCG@20'],
                    'MRR@5': results['test_metrics']['MRR@5'],
                    'MRR@10': results['test_metrics']['MRR@10'],
                    'MRR@20': results['test_metrics']['MRR@20'],
                    
                    # Short sequence metrics
                    'Short_HR@10': results['grouped_metrics']['short']['HR@10'],
                    'Short_NDCG@10': results['grouped_metrics']['short']['NDCG@10'],
                    'Short_MRR@10': results['grouped_metrics']['short']['MRR@10'],
                    'Short_Count': results['grouped_metrics']['short']['count'],
                    
                    # Medium sequence metrics
                    'Medium_HR@10': results['grouped_metrics']['medium']['HR@10'],
                    'Medium_NDCG@10': results['grouped_metrics']['medium']['NDCG@10'],
                    'Medium_MRR@10': results['grouped_metrics']['medium']['MRR@10'],
                    'Medium_Count': results['grouped_metrics']['medium']['count'],
                    
                    # Training info
                    'Best_Epoch': results['best_epoch'],
                    'Best_Val_NDCG@10': results['best_val_metric'],
                    
                    # Config
                    'd_model': config.get('d_model', 64),
                    'n_heads': config.get('n_heads', 2),
                    'n_blocks': config.get('n_blocks', 2),
                    'gnn_layers': config.get('gnn_layers', 2),
                    'dropout': config.get('dropout', 0.2),
                    'batch_size': config.get('batch_size', 256),
                    'lr': config.get('lr', 0.001),
                }
                
                all_results.append(result_data)
                
                # Store model name for later reference
                per_user_metrics[model_name] = result_folder

# Create DataFrame
df_results = pd.DataFrame(all_results)

# Sort by NDCG@10 (descending)
df_results = df_results.sort_values('NDCG@10', ascending=False).reset_index(drop=True)

print(f"\n✅ Collected results from {len(df_results)} experiments")
print("\nModels found:")
print("\nBASELINES:")
for model in df_results['Model'].values:
    print(f"  - {model}")

print("\n" + "="*80)
print("📈 OVERALL PERFORMANCE RANKING (by NDCG@10)")
print("="*80)
print(df_results[['Model', 'Model_Type', 'NDCG@10', 'HR@10', 'MRR@10', 'Best_Epoch']].to_string(index=False))
print("="*80)

## MARKDOWN CELL 12
## Step 6: Baseline Model Comparison

Compare performance across all baseline models

## CODE CELL 13
import numpy as np

print("="*80)
print("📊 BASELINE MODEL COMPARISON")
print("="*80)

# Get baselines
baselines = df_results[['Model', 'NDCG@10', 'HR@10', 'MRR@10']].copy()

print("\n" + "="*80)
print("BASELINE MODELS PERFORMANCE")
print("="*80)
print(baselines.to_string(index=False))
print("="*80)

# Find best and worst performers
best_model = baselines.iloc[0]
worst_model = baselines.iloc[-1]

print(f"\n🏆 Best performing baseline: {best_model['Model']}")
print(f"   NDCG@10: {best_model['NDCG@10']:.6f}")
print(f"   HR@10: {best_model['HR@10']:.6f}")
print(f"   MRR@10: {best_model['MRR@10']:.6f}")

print(f"\n📊 Lowest performing baseline: {worst_model['Model']}")
print(f"   NDCG@10: {worst_model['NDCG@10']:.6f}")
print(f"   HR@10: {worst_model['HR@10']:.6f}")
print(f"   MRR@10: {worst_model['MRR@10']:.6f}")

# Calculate performance gap
gap_ndcg = ((best_model['NDCG@10'] - worst_model['NDCG@10']) / worst_model['NDCG@10']) * 100
gap_hr = ((best_model['HR@10'] - worst_model['HR@10']) / worst_model['HR@10']) * 100

print(f"\n📈 Performance gap (best vs worst):")
print(f"   NDCG@10: {gap_ndcg:+.2f}%")
print(f"   HR@10: {gap_hr:+.2f}%")

# Compare each model to the best
print("\n" + "="*80)
print("COMPARISON TO BEST BASELINE")
print("="*80)
for _, row in baselines.iterrows():
    if row['Model'] == best_model['Model']:
        print(f"{row['Model']:15s}: BEST PERFORMER")
    else:
        ndcg_diff = ((row['NDCG@10'] - best_model['NDCG@10']) / best_model['NDCG@10']) * 100
        hr_diff = ((row['HR@10'] - best_model['HR@10']) / best_model['HR@10']) * 100
        print(f"{row['Model']:15s}: {ndcg_diff:+6.2f}% NDCG@10, {hr_diff:+6.2f}% HR@10")

print("="*80)

# Create publication-ready table
print("\n📋 PUBLICATION TABLE FORMAT:")
print("-"*100)
print("Model                          | Type     | NDCG@10      | HR@10        | MRR@10       | Epoch")
print("-"*100)
for idx, row in df_results.iterrows():
    model = row['Model']
    model_type = row['Model_Type']
    ndcg = row['NDCG@10']
    hr = row['HR@10']
    mrr = row['MRR@10']
    epoch = row['Best_Epoch']
    
    print(f"{model:30s} | {model_type:8s} | {ndcg:.6f}     | {hr:.6f}     | {mrr:.6f}     | {epoch:3d}")

print("-"*100)
print("="*80)

## MARKDOWN CELL 14
## Step 7: Performance by Sequence Length

Detailed analysis of short vs medium sequence performance

## CODE CELL 15
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

print("="*80)
print("👥 PERFORMANCE BY SEQUENCE LENGTH")
print("="*80)

# Short sequences
print("\nSHORT SEQUENCES (<10 items, n=162 users):")
print("-"*80)
df_short = df_results[['Model', 'Model_Type', 'Short_NDCG@10', 'Short_HR@10', 'Short_MRR@10']].copy()
df_short.columns = ['Model', 'Type', 'NDCG@10', 'HR@10', 'MRR@10']
print(df_short.sort_values('NDCG@10', ascending=False).to_string(index=False))

# Get best baseline for short
best_baseline_short = df_results.sort_values('Short_NDCG@10', ascending=False).iloc[0]
worst_baseline_short = df_results.sort_values('Short_NDCG@10', ascending=True).iloc[0]
baseline_short_ndcg = best_baseline_short['Short_NDCG@10']
print(f"\nBest Baseline Short NDCG@10: {best_baseline_short['Model']} ({baseline_short_ndcg:.6f})")
print(f"Worst Baseline Short NDCG@10: {worst_baseline_short['Model']} ({worst_baseline_short['Short_NDCG@10']:.6f})")

# Medium sequences
print("\n" + "="*80)
print("MEDIUM SEQUENCES (10-50 items, n=5,872 users):")
print("-"*80)
df_medium = df_results[['Model', 'Model_Type', 'Medium_NDCG@10', 'Medium_HR@10', 'Medium_MRR@10']].copy()
df_medium.columns = ['Model', 'Type', 'NDCG@10', 'HR@10', 'MRR@10']
print(df_medium.sort_values('NDCG@10', ascending=False).to_string(index=False))

# Get best baseline for medium
best_baseline_medium = df_results.sort_values('Medium_NDCG@10', ascending=False).iloc[0]
worst_baseline_medium = df_results.sort_values('Medium_NDCG@10', ascending=True).iloc[0]
baseline_medium_ndcg = best_baseline_medium['Medium_NDCG@10']
print(f"\nBest Baseline Medium NDCG@10: {best_baseline_medium['Model']} ({baseline_medium_ndcg:.6f})")
print(f"Worst Baseline Medium NDCG@10: {worst_baseline_medium['Model']} ({worst_baseline_medium['Medium_NDCG@10']:.6f})")

print("="*80)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: NDCG@10 by sequence length
models = df_results['Model'].values
short_ndcg = df_results['Short_NDCG@10'].values
medium_ndcg = df_results['Medium_NDCG@10'].values
overall_ndcg = df_results['NDCG@10'].values

# Sort by overall NDCG
sort_idx = np.argsort(overall_ndcg)[::-1]
models = models[sort_idx]
short_ndcg = short_ndcg[sort_idx]
medium_ndcg = medium_ndcg[sort_idx]
overall_ndcg = overall_ndcg[sort_idx]

x = np.arange(len(models))
width = 0.25

axes[0].bar(x - width, short_ndcg, width, label='Short (<10)', color='steelblue', alpha=0.7)
axes[0].bar(x, medium_ndcg, width, label='Medium (10-50)', color='navy', alpha=0.7)
axes[0].bar(x + width, overall_ndcg, width, label='Overall', color='gray', alpha=0.7)
axes[0].set_xlabel('Model')
axes[0].set_ylabel('NDCG@10')
axes[0].set_title('NDCG@10 by Sequence Length (Baseline Models)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Performance gap analysis (short vs medium)
performance_gaps = []
for idx, row in df_results.iterrows():
    gap = ((row['Medium_NDCG@10'] - row['Short_NDCG@10']) / row['Short_NDCG@10']) * 100
    performance_gaps.append(gap)

x2 = np.arange(len(models))
colors = ['green' if g > 0 else 'red' for g in performance_gaps]
axes[1].bar(x2, performance_gaps, alpha=0.7, color=colors)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Performance Gap (%)')
axes[1].set_title('Medium vs Short Sequence Performance Gap (positive = better on medium)')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(models, rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/performance_by_length.png', dpi=300, bbox_inches='tight')
print("\n✅ Figure saved: results/performance_by_length.png")
plt.show()

## MARKDOWN CELL 16
## Step 8: Training Dynamics Analysis

Analyze convergence speed, best epochs, and validation performance

## CODE CELL 17
print("="*80)
print("📈 TRAINING DYNAMICS ANALYSIS")
print("="*80)

# Training statistics
training_stats = df_results[['Model', 'Best_Epoch', 'Best_Val_NDCG@10']].copy()
training_stats = training_stats.sort_values('Best_Epoch')

print("\nCONVERGENCE ANALYSIS (sorted by convergence speed):")
print("-"*80)
print(training_stats.to_string(index=False))
print("-"*80)

# Analyze convergence patterns
fastest = training_stats.iloc[0]
slowest = training_stats.iloc[-1]
print(f"\nFastest convergence: {fastest['Model']} (epoch {fastest['Best_Epoch']})")
print(f"Slowest convergence: {slowest['Model']} (epoch {slowest['Best_Epoch']})")
print(f"Average convergence: {training_stats['Best_Epoch'].mean():.1f} epochs")

# Best validation performance
best_val = df_results.sort_values('Best_Val_NDCG@10', ascending=False).iloc[0]
print(f"\nBest validation NDCG@10: {best_val['Model']} ({best_val['Best_Val_NDCG@10']:.6f})")

print("\n" + "="*80)
print("TRAINING CONFIGURATION (consistent across all models):")
print("-"*80)
config_info = df_results.iloc[0]
print(f"  Max epochs: 200")
print(f"  Early stopping patience: 20")
print(f"  Batch size: {config_info['batch_size']}")
print(f"  Learning rate: {config_info['lr']}")
print(f"  Embedding dimension (d_model): {config_info['d_model']}")
print(f"  Attention heads (n_heads): {config_info['n_heads']}")
print(f"  Transformer blocks (n_blocks): {config_info['n_blocks']}")
print(f"  GNN layers: {config_info['gnn_layers']} (for hybrid models)")
print(f"  Dropout: {config_info['dropout']}")
print("="*80)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Best epoch vs model
models = training_stats['Model'].values
epochs = training_stats['Best_Epoch'].values
axes[0].barh(models, epochs, alpha=0.7, color='steelblue')
axes[0].set_xlabel('Best Epoch')
axes[0].set_ylabel('Model')
axes[0].set_title('Convergence Speed (Lower is Faster)')
axes[0].grid(axis='x', alpha=0.3)

# Plot 2: Validation vs test performance correlation
val_perf = df_results['Best_Val_NDCG@10'].values
test_perf = df_results['NDCG@10'].values
axes[1].scatter(val_perf, test_perf, s=100, alpha=0.7)
for i, model in enumerate(df_results['Model'].values):
    axes[1].annotate(model, (val_perf[i], test_perf[i]), 
                    fontsize=8, ha='right', va='bottom')
axes[1].plot([min(val_perf), max(val_perf)], [min(val_perf), max(val_perf)], 
             'r--', alpha=0.5, label='Perfect correlation')
axes[1].set_xlabel('Validation NDCG@10')
axes[1].set_ylabel('Test NDCG@10')
axes[1].set_title('Validation vs Test Performance')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_dynamics.png', dpi=300, bbox_inches='tight')
print("\n✅ Figure saved: results/training_dynamics.png")
plt.show()

## MARKDOWN CELL 18
## Step 9: Computational Cost Analysis

Analyze parameters, memory usage, and training efficiency

## CODE CELL 19
import torch
import sys
sys.path.insert(0, '/kaggle/working/length-adaptive')

from src.models.bert4rec import BERT4Rec
from src.models.sasrec import SASRec
from src.models.gru4rec import GRU4Rec
from src.models.caser import Caser
from src.models.lightgcn import LightGCN

print("="*80)
print("💻 COMPUTATIONAL COST ANALYSIS")
print("="*80)

# Model configurations
num_items = 3706  # MovieLens-1M
d_model = 64
n_heads = 2
n_blocks = 2
gnn_layers = 2
max_len = 50

computational_costs = []

# BERT4Rec baseline
print("\nAnalyzing BERT4Rec...")
model = BERT4Rec(num_items, d_model, n_heads, n_blocks, max_len=max_len)
bert_params = sum(p.numel() for p in model.parameters())
bert_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
computational_costs.append({
    'Model': 'bert4rec',
    'Model_Type': 'Baseline',
    'Total_Params': bert_params,
    'Trainable_Params': bert_trainable,
    'Params_M': bert_params / 1e6
})

# SASRec baseline
print("Analyzing SASRec...")
model = SASRec(num_items, d_model, n_heads, n_blocks, max_len=max_len)
sas_params = sum(p.numel() for p in model.parameters())
sas_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
computational_costs.append({
    'Model': 'sasrec',
    'Model_Type': 'Baseline',
    'Total_Params': sas_params,
    'Trainable_Params': sas_trainable,
    'Params_M': sas_params / 1e6
})

# GRU4Rec baseline
print("Analyzing GRU4Rec...")
model = GRU4Rec(num_items, d_model, n_blocks)
gru_params = sum(p.numel() for p in model.parameters())
gru_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
computational_costs.append({
    'Model': 'gru4rec',
    'Model_Type': 'Baseline',
    'Total_Params': gru_params,
    'Trainable_Params': gru_trainable,
    'Params_M': gru_params / 1e6
})

# Caser baseline
print("Analyzing Caser...")
model = Caser(num_items, d_model, max_len=max_len)
caser_params = sum(p.numel() for p in model.parameters())
caser_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
computational_costs.append({
    'Model': 'caser',
    'Model_Type': 'Baseline',
    'Total_Params': caser_params,
    'Trainable_Params': caser_trainable,
    'Params_M': caser_params / 1e6
})

# LightGCN baseline
print("Analyzing LightGCN...")
from torch_geometric.data import Data
# Create dummy edge index for parameter count
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
model = LightGCN(num_items, d_model, gnn_layers)
gcn_params = sum(p.numel() for p in model.parameters())
gcn_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
computational_costs.append({
    'Model': 'lightgcn',
    'Model_Type': 'Baseline',
    'Total_Params': gcn_params,
    'Trainable_Params': gcn_trainable,
    'Params_M': gcn_params / 1e6
})

df_costs = pd.DataFrame(computational_costs)

# Add training time from results
df_costs = df_costs.merge(
    df_results[['Model', 'Best_Epoch']], 
    on='Model', 
    how='left'
)

print("\n" + "="*80)
print("MODEL SIZE COMPARISON")
print("="*80)
print("\nBASELINES:")
print(df_costs[['Model', 'Total_Params', 'Params_M']].to_string(index=False))

print("\n" + "="*80)
print("PARAMETER BREAKDOWN:")
print("-"*80)
for idx, row in df_costs.iterrows():
    model = row['Model']
    params = row['Total_Params']
    print(f"{model:30s}: {params:>10,} params")

print("="*80)

# Merge with actual performance
df_costs = df_costs.merge(
    df_results[['Model', 'NDCG@10']], 
    on='Model', 
    how='left'
)

# Calculate efficiency metric: Performance per million parameters
df_costs['Efficiency'] = df_costs['NDCG@10'] / df_costs['Params_M']

print("\nEFFICIENCY ANALYSIS (NDCG@10 per million parameters):")
print("-"*80)
df_efficiency = df_costs[['Model', 'Model_Type', 'NDCG@10', 'Params_M', 'Efficiency']].copy()
df_efficiency = df_efficiency.sort_values('Efficiency', ascending=False)
print(df_efficiency.to_string(index=False))
print("="*80)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Parameters comparison
models = df_costs['Model'].values
params_m = df_costs['Params_M'].values

# Sort by parameters
sort_idx = np.argsort(params_m)
axes[0].barh(range(len(models)), params_m[sort_idx], color='steelblue', alpha=0.7)
axes[0].set_yticks(range(len(models)))
axes[0].set_yticklabels(models[sort_idx])
axes[0].set_xlabel('Parameters (millions)')
axes[0].set_title('Model Size Comparison (Baseline Models)')
axes[0].grid(axis='x', alpha=0.3)

# Plot 2: Performance vs parameters
ndcg = df_costs['NDCG@10'].values

axes[1].scatter(params_m, ndcg, s=150, alpha=0.7, c='steelblue', marker='o')

for i, model in enumerate(models):
    axes[1].annotate(model, (params_m[i], ndcg[i]), 
                    fontsize=9, ha='right', va='bottom')

axes[1].set_xlabel('Parameters (millions)')
axes[1].set_ylabel('NDCG@10')
axes[1].set_title('Performance vs Model Size (Baseline Models)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/computational_costs.png', dpi=300, bbox_inches='tight')
print("\n✅ Figure saved: results/computational_costs.png")
plt.show()

## MARKDOWN CELL 20
## Step 10: Export Results for Paper

Generate publication-ready tables in multiple formats (CSV, LaTeX, Markdown)

## CODE CELL 21
import os

print("="*80)
print("📝 EXPORTING PUBLICATION-READY TABLES")
print("="*80)

# Create export directory
os.makedirs('paper_results', exist_ok=True)

# 1. Main results table (baseline models only)
main_results = df_results[['Model', 'Model_Type', 'HR@5', 'HR@10', 'HR@20', 
                           'NDCG@5', 'NDCG@10', 'NDCG@20',
                           'MRR@5', 'MRR@10', 'MRR@20', 'Best_Epoch']].copy()

# Save as CSV
main_results.to_csv('paper_results/table1_main_results.csv', index=False, float_format='%.6f')
print("✅ Saved: paper_results/table1_main_results.csv")

# Generate LaTeX table
latex_main = main_results.to_latex(index=False, float_format='%.4f', 
                                   caption='Overall Performance Comparison (Baseline Models)',
                                   label='tab:main_results')
with open('paper_results/table1_main_results.tex', 'w') as f:
    f.write(latex_main)
print("✅ Saved: paper_results/table1_main_results.tex")

# 2. Sequence length analysis
length_analysis = pd.DataFrame({
    'Model': df_results['Model'],
    'Type': df_results['Model_Type'],
    'Short_HR@10': df_results['Short_HR@10'],
    'Short_NDCG@10': df_results['Short_NDCG@10'],
    'Short_MRR@10': df_results['Short_MRR@10'],
    'Medium_HR@10': df_results['Medium_HR@10'],
    'Medium_NDCG@10': df_results['Medium_NDCG@10'],
    'Medium_MRR@10': df_results['Medium_MRR@10'],
})

length_analysis.to_csv('paper_results/table2_length_analysis.csv', index=False, float_format='%.6f')
print("✅ Saved: paper_results/table2_length_analysis.csv")

latex_length = length_analysis.to_latex(index=False, float_format='%.4f',
                                       caption='Performance by Sequence Length (Baseline Models)',
                                       label='tab:length_analysis')
with open('paper_results/table2_length_analysis.tex', 'w') as f:
    f.write(latex_length)
print("✅ Saved: paper_results/table2_length_analysis.tex")

# 3. Computational costs
comp_results = df_costs[['Model', 'Model_Type', 'Total_Params', 'Params_M', 
                         'Best_Epoch', 'NDCG@10', 'Efficiency']].copy()

comp_results.to_csv('paper_results/table3_computational_costs.csv', index=False, float_format='%.6f')
print("✅ Saved: paper_results/table3_computational_costs.csv")

latex_costs = comp_results.to_latex(index=False, float_format='%.4f',
                                    caption='Computational Cost Analysis (Baseline Models)',
                                    label='tab:computational_costs')
with open('paper_results/table3_computational_costs.tex', 'w') as f:
    f.write(latex_costs)
print("✅ Saved: paper_results/table3_computational_costs.tex")

# 4. Create summary markdown
best_model = df_results.iloc[0]
worst_model = df_results.iloc[-1]

summary_md = f"""# Paper Results Summary - Baseline Models

## Dataset
- **Name:** MovieLens-1M
- **Users:** 6,040
- **Items:** 3,706
- **Interactions:** 1,000,209
- **Test Users:** 6,034
- **Short Sequences (<10):** 162 users (2.7%)
- **Medium Sequences (10-50):** 5,872 users (97.3%)

## Models Evaluated (5 baselines)

### Baselines
1. BERT4Rec (bidirectional transformer)
2. SASRec (unidirectional transformer)
3. GRU4Rec (RNN-based)
4. Caser (CNN-based)
5. LightGCN (pure GNN)

## Best Results

### Overall Performance (NDCG@10)
- **Best Baseline:** {best_model['Model']} ({best_model['NDCG@10']:.6f})
- **Worst Baseline:** {worst_model['Model']} ({worst_model['NDCG@10']:.6f})

### Performance Gap
- **NDCG@10:** {((best_model['NDCG@10'] - worst_model['NDCG@10']) / worst_model['NDCG@10'] * 100):.2f}%
- **HR@10:** {((best_model['HR@10'] - worst_model['HR@10']) / worst_model['HR@10'] * 100):.2f}%
- **MRR@10:** {((best_model['MRR@10'] - worst_model['MRR@10']) / worst_model['MRR@10'] * 100):.2f}%

## Key Findings

1. **Best Performing Model:** {best_model['Model']}
   - NDCG@10: {best_model['NDCG@10']:.6f}
   - HR@10: {best_model['HR@10']:.6f}
   - MRR@10: {best_model['MRR@10']:.6f}

2. **Parameter Efficiency:** 
   - Smallest model: {df_costs.loc[df_costs['Total_Params'].idxmin(), 'Model']} ({df_costs['Total_Params'].min():,} parameters)
   - Largest model: {df_costs.loc[df_costs['Total_Params'].idxmax(), 'Model']} ({df_costs['Total_Params'].max():,} parameters)

3. **Convergence Speed:**
   - Average best epoch: {df_results['Best_Epoch'].mean():.1f}
   - Range: {df_results['Best_Epoch'].min()}-{df_results['Best_Epoch'].max()} epochs

## Files Generated

- `table1_main_results` - Overall performance metrics (5 baseline models)
- `table1_main_results` - Overall performance metrics (4 baseline models)
- `table2_length_analysis` - Performance by sequence length
- `table3_computational_costs` - Model size and efficiency

### Figures (PNG, 300 DPI)
- `performance_by_length.png` - NDCG@10 comparison by sequence length
- `training_dynamics.png` - Convergence and validation analysis
- `computational_costs.png` - Parameter and efficiency analysis

### Raw Data
- All experimental results in results/ directory
- Per-model config, history, and results JSON files
"""

with open('paper_results/README.md', 'w') as f:
    f.write(summary_md)
print("✅ Saved: paper_results/README.md")

print("\n" + "="*80)
print("✅ ALL TABLES AND FIGURES EXPORTED")
print("="*80)
print("\nGenerated files:")
print("  📊 3 CSV files (Excel-compatible)")
print("  📄 3 LaTeX files (ready for paper)")
print("  📈 3 PNG figures (300 DPI, publication quality)")
print("  📝 1 Summary README")
print("\nLocation: paper_results/")
print("="*80)

## MARKDOWN CELL 22
## Step 11: Create Downloadable Archive

Package all results, figures, and tables for easy download

## CODE CELL 23
# Create comprehensive results archive
!mkdir -p final_package

# Copy all paper-ready materials
!cp -r paper_results final_package/
!cp -r results/*.png final_package/ 2>/dev/null || true
!cp results/*_comparison_*.csv final_package/paper_results/ 2>/dev/null || true

# Copy experimental results (baseline models only)
!mkdir -p final_package/raw_results
!cp -r results/bert4rec_* final_package/raw_results/ 2>/dev/null || true
!cp -r results/sasrec_* final_package/raw_results/ 2>/dev/null || true
!cp -r results/gru4rec_* final_package/raw_results/ 2>/dev/null || true
!cp -r results/caser_* final_package/raw_results/ 2>/dev/null || true
!cp -r results/lightgcn_* final_package/raw_results/ 2>/dev/null || true

# Create archive
!zip -r paper_results_package.zip final_package/

print("="*80)
print("📦 FINAL PACKAGE CREATED")
print("="*80)
print("\n✅ File: paper_results_package.zip")
print("\nContents:")
print("  📁 paper_results/")
print("     ├── table1_main_results.csv + .tex (4 baseline models)")
print("     ├── table2_length_analysis.csv + .tex")
print("     ├── table3_computational_costs.csv + .tex")
print("     ├── performance_by_length.png")
print("     ├── training_dynamics.png")
print("     ├── computational_costs.png")
print("     └── README.md")
print("  📁 raw_results/")
print("     ├── bert4rec_*/")
print("     ├── sasrec_*/")
print("     ├── gru4rec_*/")
print("     └── lightgcn_*/")
print("")
print("📥 Download this file to use in your paper!")
print("="*80)

# Display file size
import os
file_size = os.path.getsize('paper_results_package.zip') / 1024 / 1024

print(f"\nPackage size: {file_size:.2f} MB")print(f"\nPackage size: {file_size:.2f} MB")

## MARKDOWN CELL 24
## 📋 Final Summary

### ✅ What Was Collected:

1. **Performance Metrics**
   - Overall test performance (HR@5/10/20, NDCG@5/10/20, MRR@5/10/20)
   - Performance by sequence length (short vs medium)
   - Baseline model comparisons

2. **Training Analysis**
   - Convergence speed (best epochs)
   - Validation performance tracking
   - Training configuration details

3. **Computational Costs**
   - Model parameters (total and trainable) for all 5 baseline models
   - Parameter comparison analysis
   - Efficiency metrics (performance per parameter)

4. **Publication Materials**
   - CSV tables (Excel-compatible)
   - LaTeX tables (ready for paper)
   - High-resolution figures (300 DPI PNG)
   - Summary documentation

### 📊 Models Compared:

**Baselines (5):**
- BERT4Rec
- SASRec
- GRU4Rec
- Caser
- LightGCN

### 📥 Next Steps:

1. Download `paper_results_package.zip`
2. Review all tables and figures
3. Use LaTeX tables directly in your paper
4. Include figures with captions
5. Use these baseline results to compare with hybrid models

**Ready for paper submission!** 🎉

## CODE CELL 25
!zip -r /kaggle/working/.virtual_documents/length-adaptive.zip /kaggle/working/length-adaptive
