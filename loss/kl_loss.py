import torch

class SmoothedKLDivergenceLoss(torch.nn.Module):
    """
    Custom KL Divergence Loss Module.
    """

    def __init__(self, num_classes, epsilon=0.5):
        super(SmoothedKLDivergenceLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, predicted_probs, target_probs):
        """
        Computes the KL Divergence loss between predicted probabilities (Q) and target probabilities (P).

        Args:
            predicted_probs (torch.Tensor): Predicted probabilities, shape (batch_size, num_classes).
            target_probs (torch.Tensor): Target probabilities, shape (batch_size, num_classes).

        Returns:
            torch.Tensor: The KL divergence loss averaged over the batch.
        """
        # Ensure probabilities are non-zero to avoid log(0)
        small_epsilon = 1e-10
        predicted_probs = torch.clamp(predicted_probs, min=small_epsilon, max=1.0)
        target_probs = torch.clamp(target_probs, min=small_epsilon, max=1.0)

        soft_target_probs = (1 - self.epsilon) * target_probs + self.epsilon / self.num_classes

        # Compute element-wise KL divergence for each sample
        kl_divergences = soft_target_probs * (torch.log(soft_target_probs) - torch.log(predicted_probs))

        # Sum over classes and then average over the batch
        batch_kl_loss = torch.sum(kl_divergences, dim=1)  # Sum over classes
        average_kl_loss = torch.mean(batch_kl_loss)  # Average over the batch

        return average_kl_loss