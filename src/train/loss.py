import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss"""

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: [batch_size] - scores for positive items
            neg_scores: [batch_size, num_neg] - scores for negative items

        Returns:
            loss: scalar
        """
        # BPR: -log(sigmoid(pos - neg))
        # Equivalent to: log(1 + exp(neg - pos))
        diff = neg_scores - pos_scores.unsqueeze(1)
        loss = F.softplus(diff).mean()
        return loss


class BCELoss(nn.Module):
    """Binary Cross-Entropy loss"""

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: [batch_size]
            neg_scores: [batch_size, num_neg]

        Returns:
            loss: scalar
        """
        # Positive loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores,
            torch.ones_like(pos_scores)
        )

        # Negative loss
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores,
            torch.zeros_like(neg_scores)
        )

        return pos_loss + neg_loss


class SampledSoftmaxLoss(nn.Module):
    """Sampled softmax loss (treats pos + neg as classification)"""

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: [batch_size]
            neg_scores: [batch_size, num_neg]

        Returns:
            loss: scalar
        """
        # Combine pos and neg scores
        # Shape: [batch_size, 1 + num_neg]
        scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        # Target is always index 0 (positive item)
        targets = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)

        # Cross entropy loss
        loss = F.cross_entropy(scores, targets)
        return loss


# Testing
if __name__ == '__main__':
    batch_size = 4
    num_neg = 3

    # Dummy scores (pos should be higher than neg)
    pos_scores = torch.randn(batch_size)
    neg_scores = torch.randn(batch_size, num_neg) - 1.0  # Make negatives lower

    print("Testing Loss Functions:")
    print("="*50)

    # Test BPR Loss
    print("\n1. BPR Loss:")
    bpr_loss_fn = BPRLoss()
    loss = bpr_loss_fn(pos_scores, neg_scores)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   (Lower is better when pos > neg)")

    # Test BCE Loss
    print("\n2. BCE Loss:")
    bce_loss_fn = BCELoss()
    loss = bce_loss_fn(pos_scores, neg_scores)
    print(f"   Loss: {loss.item():.4f}")

    # Test Sampled Softmax
    print("\n3. Sampled Softmax Loss:")
    softmax_loss_fn = SampledSoftmaxLoss()
    loss = softmax_loss_fn(pos_scores, neg_scores)
    print(f"   Loss: {loss.item():.4f}")

    # Test gradient flow
    print("\n4. Gradient Flow Test:")
    pos_scores = torch.randn(batch_size, requires_grad=True)
    neg_scores = torch.randn(batch_size, num_neg, requires_grad=True)

    loss = bpr_loss_fn(pos_scores, neg_scores)
    loss.backward()

    print(f"   Loss: {loss.item():.4f}")
    print(f"   Pos grad norm: {pos_scores.grad.norm().item():.4f}")
    print(f"   Neg grad norm: {neg_scores.grad.norm().item():.4f}")

    print("\n✅ All loss functions working!")
    print("="*50)
