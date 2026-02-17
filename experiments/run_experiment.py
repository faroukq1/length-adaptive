"""
Main experiment runner for Length-Adaptive Sequential Recommendation

Usage:
    python experiments/run_experiment.py --model hybrid_discrete --epochs 50
    python experiments/run_experiment.py --model sasrec --epochs 50
    python experiments/run_experiment.py --help
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

from src.models.sasrec import SASRec
from src.models.hybrid import HybridSASRecGNN
from src.data.dataloader import get_dataloaders
from src.train.trainer import Trainer
from src.train.loss import BPRLoss

def create_model(model_type, num_items, args):
    """Create model based on type"""
    
    if model_type == 'sasrec':
        model = SASRec(
            num_items=num_items,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            d_ff=args.d_ff,
            max_len=args.max_len,
            dropout=args.dropout
        )
    
    elif model_type in ['hybrid_fixed', 'hybrid_discrete', 'hybrid_learnable', 'hybrid_continuous']:
        fusion_type = model_type.replace('hybrid_', '')
        
        model = HybridSASRecGNN(
            num_items=num_items,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            d_ff=args.d_ff,
            max_len=args.max_len,
            gnn_layers=args.gnn_layers,
            dropout=args.dropout,
            fusion_type=fusion_type,
            fixed_alpha=args.fixed_alpha,
            L_short=args.L_short,
            L_long=args.L_long
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main(args):
    """Main experiment function"""
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n[1/5] Loading preprocessed data...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    num_items = data['config']['num_items']
    print(f"  Users: {data['config']['num_users']:,}")
    print(f"  Items: {num_items:,}")
    
    # Load graph
    print("\n[2/5] Loading co-occurrence graph...")
    with open(args.graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    edge_index = graph_data['edge_index']
    edge_weight = graph_data['edge_weight']
    print(f"  Edges: {edge_index.shape[1]:,}")
    
    # Create dataloaders
    print("\n[3/5] Creating dataloaders...")
    train_loader, val_loader, test_loader, config = get_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\n[4/5] Creating model: {args.model}")
    model = create_model(args.model, num_items, args)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.model}_{timestamp}"
    exp_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment config
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"  Experiment dir: {exp_dir}")
    
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
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    history = trainer.train(
        num_epochs=args.epochs,
        eval_every=args.eval_every
    )
    
    # Save training history
    history_path = os.path.join(exp_dir, 'history.json')
    with open(history_path, 'w') as f:
        # Convert to JSON-serializable format
        history_json = {
            'train_loss': history['train_loss'],
            'val_metrics': history['val_metrics'],
            'best_epoch': history['best_epoch'],
            'best_val_metric': history['best_val_metric']
        }
        json.dump(history_json, f, indent=2)
    
    # Test
    print("\n" + "="*60)
    print("TESTING")
    print("="*60)
    test_metrics, grouped_metrics = trainer.test(use_best_model=True)
    
    # Save test results
    results = {
        'test_metrics': test_metrics,
        'grouped_metrics': grouped_metrics,
        'best_epoch': history['best_epoch'],
        'best_val_metric': history['best_val_metric']
    }
    
    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Experiment complete! Results saved to: {exp_dir}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sequential recommendation experiment')
    
    # Model selection
    parser.add_argument('--model', type=str, default='hybrid_discrete',
                       choices=['sasrec', 'hybrid_fixed', 'hybrid_discrete', 
                               'hybrid_learnable', 'hybrid_continuous'],
                       help='Model type to train')
    
    # Data paths
    parser.add_argument('--data_path', type=str, 
                       default='data/ml-1m/processed/ml1m_sequential.pkl',
                       help='Path to processed data')
    parser.add_argument('--graph_path', type=str,
                       default='data/graphs/cooccurrence_graph.pkl',
                       help='Path to co-occurrence graph')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=2,
                       help='Number of attention heads')
    parser.add_argument('--n_blocks', type=int, default=2,
                       help='Number of transformer blocks')
    parser.add_argument('--d_ff', type=int, default=256,
                       help='Feed-forward dimension')
    parser.add_argument('--gnn_layers', type=int, default=2,
                       help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--max_len', type=int, default=50,
                       help='Maximum sequence length')
    
    # Fusion parameters
    parser.add_argument('--fixed_alpha', type=float, default=0.5,
                       help='Fixed alpha for fusion (if using fixed fusion)')
    parser.add_argument('--L_short', type=int, default=10,
                       help='Threshold for short history users')
    parser.add_argument('--L_long', type=int, default=50,
                       help='Threshold for long history users')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU even if GPU available')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run experiment
    results = main(args)
