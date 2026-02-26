"""
Run Best Models: BERT Hybrid (Fixed & Discrete) + TCN-BERT4Rec
Trains the top-performing models for publication-quality results
"""

import argparse
import pickle
import torch
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bert4rec_hybrid import HybridBERT4RecGNN
from src.models.tcn_bert4rec import TCNBERT4Rec
from src.models.tgt_bert4rec import TGT_BERT4Rec
from src.data.dataloader import get_dataloaders
from src.train.trainer import Trainer


def create_model(model_type, num_items, args):
    """Create model based on type"""
    
    if model_type == 'bert_hybrid_fixed':
        model = HybridBERT4RecGNN(
            num_items=num_items,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            d_ff=args.d_ff,
            max_len=args.max_len,
            gnn_layers=args.gnn_layers,
            dropout=args.dropout,
            fusion_type='fixed',
            fixed_alpha=args.fixed_alpha
        )
    
    elif model_type == 'bert_hybrid_discrete':
        model = HybridBERT4RecGNN(
            num_items=num_items,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            d_ff=args.d_ff,
            max_len=args.max_len,
            gnn_layers=args.gnn_layers,
            dropout=args.dropout,
            fusion_type='discrete',
            L_short=args.L_short,
            L_long=args.L_long
        )
    
    elif model_type == 'tcn_bert4rec':
        model = TCNBERT4Rec(
            num_items=num_items,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            d_ff=args.d_ff,
            max_len=args.max_len,
            tcn_channels=args.tcn_channels,
            tcn_kernel_size=args.tcn_kernel_size,
            dropout=args.dropout,
            fusion_type=args.tcn_fusion
        )
    
    elif model_type == 'tgt_bert4rec':
        model = TGT_BERT4Rec(
            num_items=num_items,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            d_ff=args.d_ff,
            max_len=args.max_len,
            dropout=args.dropout,
            fusion_alpha=args.tgt_fusion_alpha,
            learnable_fusion=args.tgt_learnable_fusion
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def run_experiment(model_type, args):
    """Run single experiment"""
    
    print("\n" + "="*70)
    print(f"RUNNING EXPERIMENT: {model_type.upper()}")
    print("="*70)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\n[1/5] Loading data...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    num_items = data['config']['num_items']
    print(f"  Items: {num_items:,}")
    
    # Load graph (only for hybrid models)
    edge_index = None
    edge_weight = None
    if model_type in ['bert_hybrid_fixed', 'bert_hybrid_discrete']:
        print("\n[2/5] Loading graph...")
        with open(args.graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        edge_index = graph_data['edge_index']
        edge_weight = graph_data['edge_weight']
        print(f"  Edges: {edge_index.shape[1]:,}")
    else:
        print("\n[2/5] Skipping graph (TCN model doesn't use graph)")
    
    # Create dataloaders
    print("\n[3/5] Creating dataloaders...")
    train_loader, val_loader, test_loader, config = get_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers
    )
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Valid: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    # Create model
    print("\n[4/5] Creating model...")
    model = create_model(model_type, num_items, args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Check for existing experiment directory to resume
    exp_dir = None
    start_epoch = 0
    checkpoint_file = None
    
    if args.resume:
        # Look for existing experiment directories for this model
        import glob
        
        # Search in multiple patterns:
        # 1. results/model_type_* (timestamp-based)
        # 2. results/*/model_type (dataset subdirectory pattern like ml-1m)
        existing_dirs = sorted(glob.glob(os.path.join(args.results_dir, f"{model_type}_*")))
        existing_dirs += sorted(glob.glob(os.path.join(args.results_dir, f"*/{model_type}")))
        
        if existing_dirs:
            # Use the most recent directory (by modification time)
            exp_dir = max(existing_dirs, key=lambda x: os.path.getmtime(x))
            
            # Check for checkpoints (prefer best_model, then latest epoch checkpoint)
            best_checkpoint = os.path.join(exp_dir, 'best_model.pt')
            epoch_checkpoints = sorted(glob.glob(os.path.join(exp_dir, 'checkpoint_epoch_*.pt')))
            
            if os.path.exists(best_checkpoint):
                checkpoint_file = 'best_model.pt'
                print(f"  ✅ Found best checkpoint: {exp_dir}")
                print(f"  📥 Resuming from best model...")
            elif epoch_checkpoints:
                # Use the latest epoch checkpoint
                checkpoint_file = os.path.basename(epoch_checkpoints[-1])
                epoch_num = checkpoint_file.split('_')[-1].replace('.pt', '')
                print(f"  ✅ Found checkpoint: {exp_dir}")
                print(f"  📥 Resuming from epoch {epoch_num}...")
            else:
                print(f"  ⚠️  Directory exists but no checkpoint found")
                exp_dir = None
    
    # Create new experiment directory if not resuming
    if exp_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(args.results_dir, f"{model_type}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        print(f"  🆕 Starting new experiment: {exp_dir}")
    
    # Save configuration
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create trainer
    print("\n[5/5] Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_dir=exp_dir
    )
    
    # Load checkpoint if resuming
    start_epoch = 0  # Initialize start_epoch
    if args.resume and checkpoint_file:
        try:
            start_epoch = trainer.load_checkpoint(checkpoint_file)
            print(f"  ✅ Loaded checkpoint from epoch {start_epoch}")
            print(f"  📊 Best NDCG@10 so far: {trainer.history.get('best_val_metric', 0.0):.6f}")
        except Exception as e:
            print(f"  ⚠️  Failed to load checkpoint: {e}")
            print(f"  🔄 Starting from scratch")
            start_epoch = 0
    
    # Train
    print("\n" + "="*70)
    print("TRAINING" + (" (RESUMED)" if start_epoch > 0 else ""))
    print("="*70)
    if start_epoch > 0:
        print(f"Starting from epoch {start_epoch + 1}")
    
    history = trainer.train(
        num_epochs=args.epochs,
        eval_every=args.eval_every,
        start_epoch=start_epoch
    )
    
    # Save training history
    history_path = os.path.join(exp_dir, 'history.json')
    with open(history_path, 'w') as f:
        history_json = {
            'train_loss': history['train_loss'],
            'val_metrics': history['val_metrics'],
            'best_epoch': history['best_epoch'],
            'best_val_metric': history['best_val_metric']
        }
        json.dump(history_json, f, indent=2)
    
    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    test_metrics, grouped_metrics = trainer.test(use_best_model=True)
    
    # Save test results
    results = {
        'test_metrics': test_metrics,
        'grouped_metrics': grouped_metrics,
        'config': vars(args),
        'best_epoch': history['best_epoch']
    }
    
    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ EXPERIMENT COMPLETE: {model_type}")
    print("="*70)
    print(f"📁 Results saved to: {exp_dir}")
    print(f"📊 Best epoch: {history['best_epoch']}")
    print(f"📈 Test HR@10: {test_metrics['HR@10']:.6f}")
    print(f"📈 Test NDCG@10: {test_metrics['NDCG@10']:.6f}")
    print(f"📈 Test MRR@10: {test_metrics['MRR@10']:.6f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run Best Models Experiments')
    
    # Data
    parser.add_argument('--data_path', type=str, 
                       default='data/ml-1m/processed/sequences.pkl',
                       help='Path to processed data')
    parser.add_argument('--graph_path', type=str,
                       default='data/graphs/cooccurrence_graph.pkl',
                       help='Path to graph data')
    parser.add_argument('--results_dir', type=str, 
                       default='results',
                       help='Directory to save results')
    
    # Models to run
    parser.add_argument('--models', type=str, nargs='+',
                       default=['bert_hybrid_fixed', 'bert_hybrid_discrete', 'tcn_bert4rec', 'tgt_bert4rec'],
                       choices=['bert_hybrid_fixed', 'bert_hybrid_discrete', 'tcn_bert4rec', 'tgt_bert4rec'],
                       help='Models to train')
    
    # Model architecture (Fine-tuned optimal configuration)
    parser.add_argument('--d_model', type=int, default=64,
                       help='Embedding dimension (fine-tuned optimal)')
    parser.add_argument('--n_heads', type=int, default=2,
                       help='Number of attention heads (fine-tuned optimal)')
    parser.add_argument('--n_blocks', type=int, default=2,
                       help='Number of transformer blocks (fine-tuned optimal)')
    parser.add_argument('--d_ff', type=int, default=256,
                       help='Feed-forward dimension')
    parser.add_argument('--gnn_layers', type=int, default=2,
                       help='Number of GNN layers (for hybrid models)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (fine-tuned optimal)')
    parser.add_argument('--max_len', type=int, default=200,
                       help='Maximum sequence length')
    
    # BERT Hybrid Fixed parameters
    parser.add_argument('--fixed_alpha', type=float, default=0.5,
                       help='Fixed fusion weight for bert_hybrid_fixed')
    
    # BERT Hybrid Discrete parameters
    parser.add_argument('--L_short', type=int, default=10,
                       help='Short sequence threshold for discrete fusion')
    parser.add_argument('--L_long', type=int, default=30,
                       help='Long sequence threshold for discrete fusion')
    
    # TCN-BERT4Rec parameters
    parser.add_argument('--tcn_channels', type=int, nargs='+', 
                       default=[64, 64, 64],
                       help='TCN channel sizes')
    parser.add_argument('--tcn_kernel_size', type=int, default=3,
                       help='TCN kernel size')
    parser.add_argument('--tcn_fusion', type=str, default='learnable',
                       choices=['fixed', 'learnable', 'concat'],
                       help='TCN-BERT fusion type')
    
    # TGT-BERT4Rec parameters (Temporal Graph Transformer + BERT)
    parser.add_argument('--tgt_fusion_alpha', type=float, default=0.3,
                       help='TGT fusion weight (optimal: 0.3 for BERT bias)')
    parser.add_argument('--tgt_learnable_fusion', type=bool, default=True,
                       help='Whether TGT fusion weight is learnable')
    
    # Training (Fine-tuned optimal configuration)
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (fine-tuned optimal)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--eval_every', type=int, default=5,
                       help='Evaluate every N epochs')    
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from existing checkpoint')    
    # System
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("RUNNING BEST MODELS EXPERIMENTS")
    print("="*70)
    print(f"Models to train: {', '.join(args.models)}")
    print(f"Epochs: {args.epochs} (patience: {args.patience})")
    print(f"\nFine-Tuned Configuration:")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}, n_blocks={args.n_blocks}")
    print(f"  lr={args.lr}, dropout={args.dropout}")
    print(f"\nResults directory: {args.results_dir}")
    print("="*70)
    
    # Run experiments
    all_results = {}
    for model_type in args.models:
        try:
            results = run_experiment(model_type, args)
            all_results[model_type] = results
        except Exception as e:
            print(f"\n❌ ERROR running {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nResults Summary:")
    print(f"{'Model':<25} {'HR@10':<10} {'NDCG@10':<10} {'MRR@10':<10}")
    print("-" * 70)
    for model_type, results in all_results.items():
        metrics = results['test_metrics']
        print(f"{model_type:<25} {metrics['HR@10']:<10.6f} {metrics['NDCG@10']:<10.6f} {metrics['MRR@10']:<10.6f}")
    print("="*70)


if __name__ == '__main__':
    main()
