"""
Quick test to verify the training pipeline works

This runs a mini experiment with just 2 epochs to test:
1. Data loading works
2. Model forward pass works
3. Loss computation works
4. Backpropagation works
5. Evaluation works
6. Checkpointing works
"""

import pickle
import torch
from src.models.sasrec import SASRec
from src.models.hybrid import HybridSASRecGNN
from src.data.dataloader import get_dataloaders
from src.train.trainer import Trainer

def test_training_pipeline():
    """Test the complete training pipeline"""
    
    print("="*60)
    print("TRAINING PIPELINE TEST")
    print("="*60)
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    data_path = 'data/ml-1m/processed/sequences.pkl'
    graph_path = 'data/graphs/cooccurrence_graph.pkl'
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    num_items = data['config']['num_items']
    edge_index = graph_data['edge_index']
    edge_weight = graph_data['edge_weight']
    
    print(f"  ✓ Users: {data['config']['num_users']:,}")
    print(f"  ✓ Items: {num_items:,}")
    print(f"  ✓ Graph edges: {edge_index.shape[1]:,}")
    
    # 2. Create dataloaders
    print("\n[2/6] Creating dataloaders...")
    train_loader, val_loader, test_loader, config = get_dataloaders(
        data_path,
        batch_size=256,
        max_len=50,
        num_workers=2
    )
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches: {len(val_loader)}")
    print(f"  ✓ Test batches: {len(test_loader)}")
    
    # 3. Test SASRec model
    print("\n[3/6] Testing SASRec model...")
    sasrec = SASRec(
        num_items=num_items,
        d_model=64,
        n_heads=2,
        n_blocks=2,
        d_ff=256,
        max_len=50,
        dropout=0.2
    )
    
    # Auto-detect GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    sasrec_trainer = Trainer(
        model=sasrec,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
        lr=0.001,
        patience=5,
        save_dir='test_checkpoints/sasrec'
    )
    
    print("  Training for 2 epochs...")
    history = sasrec_trainer.train(num_epochs=2, eval_every=1)
    
    print(f"  ✓ Epoch 1 loss: {history['train_loss'][0]:.4f}")
    print(f"  ✓ Epoch 2 loss: {history['train_loss'][1]:.4f}")
    print(f"  ✓ Val NDCG@10: {history['val_metrics'][0]['ndcg@10']:.4f}")
    
    # 4. Test Hybrid model
    print("\n[4/6] Testing Hybrid model...")
    hybrid = HybridSASRecGNN(
        num_items=num_items,
        d_model=64,
        n_heads=2,
        n_blocks=2,
        d_ff=256,
        max_len=50,
        gnn_layers=2,
        dropout=0.2,
        fusion_type='discrete'
    )
    
    hybrid_trainer = Trainer(
        model=hybrid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
        lr=0.001,
        patience=5,
        save_dir='test_checkpoints/hybrid'
    )
    
    print("  Training for 2 epochs...")
    history = hybrid_trainer.train(num_epochs=2, eval_every=1)
    
    print(f"  ✓ Epoch 1 loss: {history['train_loss'][0]:.4f}")
    print(f"  ✓ Epoch 2 loss: {history['train_loss'][1]:.4f}")
    print(f"  ✓ Val NDCG@10: {history['val_metrics'][0]['ndcg@10']:.4f}")
    
    # 5. Test evaluation
    print("\n[5/6] Testing evaluation on test set...")
    test_metrics, grouped_metrics = hybrid_trainer.test(use_best_model=True)
    
    print(f"  ✓ Test HR@10: {test_metrics['hr@10']:.4f}")
    print(f"  ✓ Test NDCG@10: {test_metrics['ndcg@10']:.4f}")
    print(f"  ✓ Test MRR@10: {test_metrics['mrr@10']:.4f}")
    
    # 6. Test checkpoint loading
    print("\n[6/6] Testing checkpoint loading...")
    hybrid2 = HybridSASRecGNN(
        num_items=num_items,
        d_model=64,
        n_heads=2,
        n_blocks=2,
        d_ff=256,
        max_len=50,
        gnn_layers=2,
        dropout=0.2,
        fusion_type='discrete'
    )
    
    hybrid_trainer2 = Trainer(
        model=hybrid2,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
        lr=0.001,
        save_dir='test_checkpoints/hybrid'
    )
    
    checkpoint = hybrid_trainer2.load_checkpoint('test_checkpoints/hybrid/best_model.pt')
    print(f"  ✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  ✓ Best val metric: {checkpoint['best_val_metric']:.4f}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nThe training pipeline is working correctly!")
    print("You can now run full experiments with:")
    print("  python experiments/run_experiment.py --model hybrid_discrete --epochs 50")
    print("  python experiments/run_experiment.py --model sasrec --epochs 50")

if __name__ == '__main__':
    test_training_pipeline()
