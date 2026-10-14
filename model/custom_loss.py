import torch
import torch.nn as nn
import torch.nn.functional as F

class RiskWeightedClinicalLoss(nn.Module):
    def __init__(self, dosage_tokens: set, penalty_multiplier: float = 3.5, ignore_index: int = -100):
        super().__init__()
        self.dosage_tokens = dosage_tokens
        self.penalty = penalty_multiplier
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits shape: [batch_size, seq_len, vocab_size]
        # targets shape: [batch_size, seq_len]
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = targets.view(-1)

        raw_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=self.ignore_index, reduction="none")

        valid_mask = flat_targets != self.ignore_index
        weights = torch.ones_like(flat_targets, dtype=torch.float32)

        target_ids = flat_targets[valid_mask].tolist()
        adjusted_weights = [
            self.penalty if tid in self.dosage_tokens else 1.0
            for tid in target_ids
        ]

        if adjusted_weights:
            weights[valid_mask] = torch.tensor(adjusted_weights, device=logits.device, dtype=torch.float32)

        return (raw_loss * weights).sum() / (weights[valid_mask].sum() + 1e-8)