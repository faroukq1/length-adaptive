import torch
import torch.nn as nn
from tqdm import tqdm
import os
import time
from src.train.loss import BPRLoss
from src.eval.evaluator import Evaluator
from src.models.lightgcn_seq import LightGCNSeq

class Trainer:
    """Trainer for sequential recommendation models"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        edge_index,
        edge_weight=None,
        optimizer=None,
        criterion=None,
        device='cpu',
        lr=0.001,
        weight_decay=0.0,
        patience=10,
        save_dir='checkpoints'
    ):
        """
        Args:
            model: The recommendation model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            test_loader: Test dataloader
            edge_index: Graph edge index
            edge_weight: Graph edge weights
            optimizer: Optimizer (if None, creates Adam)
            criterion: Loss function (if None, uses BPR)
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.edge_index = edge_index.to(device) if edge_index is not None else None
        self.edge_weight = edge_weight.to(device) if edge_weight is not None else None
        self.device = device
        self.patience = patience
        self.save_dir = save_dir
        
        # Check if model uses graph
        self.use_graph = hasattr(model, 'gnn') or isinstance(model, LightGCNSeq)
        self.is_lightgcn = isinstance(model, LightGCNSeq)
        
        # Precompute graph embeddings for LightGCN (done once, reused for all batches)
        self.graph_emb = None
        if self.is_lightgcn and self.edge_index is not None:
            print("Precomputing LightGCN graph embeddings...")
            with torch.no_grad():
                self.graph_emb = self.model.compute_graph_embeddings(
                    self.edge_index, 
                    self.edge_weight
                )

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer

        # Loss function
        if criterion is None:
            self.criterion = BPRLoss()
        else:
            self.criterion = criterion

        # Evaluator
        self.evaluator = Evaluator(model, device)

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_metrics': [],
            'best_epoch': 0,
            'best_val_metric': 0.0
        }

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Move to device
            seq = batch['sequence'].to(self.device)
            lengths = batch['length'].to(self.device)
            targets = batch['target'].to(self.device)
            negatives = batch['negatives'].to(self.device)

            # Forward pass
            if self.is_lightgcn:
                # LightGCN uses precomputed graph embeddings
                seq_repr = self.model(seq, lengths, graph_emb=self.graph_emb)
            elif self.use_graph:
                # Hybrid models pass graph structure directly
                seq_repr = self.model(seq, lengths, self.edge_index, self.edge_weight)
            else:
                # Sequential models (SASRec, BERT4Rec, GRU4Rec) don't use graph
                seq_repr = self.model(seq, lengths)

            # Get scores for positive and negative items
            if self.is_lightgcn:
                pos_scores = self.model.predict(seq_repr, targets.unsqueeze(1), graph_emb=self.graph_emb).squeeze(1)
                neg_scores = self.model.predict(seq_repr, negatives, graph_emb=self.graph_emb)
            else:
                pos_scores = self.model.predict(seq_repr, targets.unsqueeze(1)).squeeze(1)
                neg_scores = self.model.predict(seq_repr, negatives)

            # Compute loss
            loss = self.criterion(pos_scores, neg_scores)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update stats
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate_epoch(self, data_loader, desc="Validating"):
        """Evaluate on validation or test set"""
        if self.is_lightgcn:
            metrics = self.evaluator.evaluate(
                data_loader,
                self.edge_index,
                self.edge_weight,
                k_list=[5, 10, 20],
                compute_by_group=False,
                verbose=True,
                graph_emb=self.graph_emb
            )
        elif self.use_graph:
            metrics = self.evaluator.evaluate(
                data_loader,
                self.edge_index,
                self.edge_weight,
                k_list=[5, 10, 20],
                compute_by_group=False,
                verbose=True
            )
        else:
            metrics = self.evaluator.evaluate(
                data_loader,
                None,
                None,
                k_list=[5, 10, 20],
                compute_by_group=False,
                verbose=True
            )
        return metrics

    def save_checkpoint(self, filename='checkpoint.pt', is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'epoch': len(self.history['train_loss'])
        }

        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, filename='checkpoint.pt'):
        """Load model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']

        return checkpoint['epoch']

    def train(self, num_epochs, eval_every=1, verbose=True, start_epoch=0):
        """
        Full training loop with early stopping

        Args:
            num_epochs: Maximum number of epochs
            eval_every: Evaluate every N epochs
            verbose: Whether to print progress
            start_epoch: Starting epoch when resuming (0-indexed)

        Returns:
            history: Training history dictionary
        """
        print("="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print("="*60 + "\n")

        # When resuming, use the loaded best_val_metric from history
        if start_epoch > 0:
            best_val_metric = self.history.get('best_val_metric', 0.0)
            patience_counter = 0
        else:
            best_val_metric = 0.0
            patience_counter = 0
        start_time = time.time()

        for epoch in range(start_epoch + 1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)

            # Evaluate
            if epoch % eval_every == 0:
                print(f"\n[Epoch {epoch}] Evaluating...")
                val_metrics = self.evaluate_epoch(self.val_loader, desc="Validating")
                self.history['val_metrics'].append(val_metrics)

                # Print results
                epoch_time = time.time() - epoch_start
                print(f"\n[Epoch {epoch}/{num_epochs}] Time: {epoch_time:.1f}s")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val HR@10: {val_metrics['HR@10']:.4f}")
                print(f"  Val NDCG@10: {val_metrics['NDCG@10']:.4f}")
                print(f"  Val MRR@10: {val_metrics['MRR@10']:.4f}")

                # Check for improvement (using NDCG@10 as primary metric)
                current_metric = val_metrics['NDCG@10']

                if current_metric > best_val_metric:
                    print(f"  ✓ New best! ({best_val_metric:.4f} → {current_metric:.4f})")
                    best_val_metric = current_metric
                    self.history['best_epoch'] = epoch
                    self.history['best_val_metric'] = best_val_metric

                    # Save best model
                    self.save_checkpoint(is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{self.patience})")

                # Early stopping
                if patience_counter >= self.patience:
                    print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                    break

            # Save regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best epoch: {self.history['best_epoch']}")
        print(f"Best val NDCG@10: {self.history['best_val_metric']:.4f}")
        print("="*60 + "\n")

        return self.history

    def test(self, use_best_model=True):
        """
        Evaluate on test set

        Args:
            use_best_model: Whether to load best model before testing

        Returns:
            test_metrics: Test set metrics
        """
        if use_best_model:
            print("Loading best model...")
            self.load_checkpoint('best_model.pt')

        print("\n" + "="*60)
        print("TESTING ON TEST SET")
        print("="*60)

        # Test overall
        if self.is_lightgcn:
            test_metrics = self.evaluator.evaluate(
                self.test_loader,
                self.edge_index,
                self.edge_weight,
                k_list=[5, 10, 20],
                compute_by_group=False,
                verbose=True,
                graph_emb=self.graph_emb
            )
        elif self.use_graph:
            test_metrics = self.evaluator.evaluate(
                self.test_loader,
                self.edge_index,
                self.edge_weight,
                k_list=[5, 10, 20],
                compute_by_group=False,
                verbose=True
            )
        else:
            test_metrics = self.evaluator.evaluate(
                self.test_loader,
                None,
                None,
                k_list=[5, 10, 20],
                compute_by_group=False,
                verbose=True
            )

        self.evaluator.print_metrics(test_metrics, "Test Results")

        # Test by group
        print("\nComputing metrics by user group...")
        if self.is_lightgcn:
            grouped_metrics = self.evaluator.evaluate(
                self.test_loader,
                self.edge_index,
                self.edge_weight,
                k_list=[10, 20],
                compute_by_group=True,
                verbose=False,
                graph_emb=self.graph_emb
            )
        elif self.use_graph:
            grouped_metrics = self.evaluator.evaluate(
                self.test_loader,
                self.edge_index,
                self.edge_weight,
                k_list=[10, 20],
                compute_by_group=True,
                verbose=False
            )
        else:
            grouped_metrics = self.evaluator.evaluate(
                self.test_loader,
                None,
                None,
                k_list=[10, 20],
                compute_by_group=True,
                verbose=False
            )

        self.evaluator.print_metrics(grouped_metrics, "Test Results by User Group")

        return test_metrics, grouped_metrics


# Testing
if __name__ == '__main__':
    import pickle
    from src.models.hybrid import HybridSASRecGNN
    from src.data.dataloader import get_dataloaders

    print("="*60)
    print("TESTING TRAINER")
    print("="*60)

    # Load data
    print("\n[1/4] Loading data...")
    with open('data/ml-1m/processed/sequences.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('data/graphs/cooccurrence_graph.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    num_items = data['config']['num_items']
    edge_index = graph_data['edge_index']
    edge_weight = graph_data['edge_weight']

    # Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    train_loader, val_loader, test_loader, config = get_dataloaders(
        'data/ml-1m/processed/sequences.pkl',
        batch_size=256,
        num_workers=0
    )

    # Create model
    print("\n[3/4] Creating model...")
    model = HybridSASRecGNN(
        num_items=num_items,
        d_model=64,
        fusion_type='discrete'
    )

    # Auto-detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create trainer
    print("\n[4/4] Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
        lr=0.001,
        patience=3,
        save_dir='test_checkpoints'
    )

    print("\n✓ Trainer created successfully!")
    print("\nTo train the model, run:")
    print("  history = trainer.train(num_epochs=50)")
    print("  test_metrics = trainer.test()")
    print("\nFor quick test (2 epochs):")
    print("  history = trainer.train(num_epochs=2)")
