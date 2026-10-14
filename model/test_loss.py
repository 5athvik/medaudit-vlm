import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from model.custom_loss import RiskWeightedClinicalLoss

dosage_ids = {42, 108, 999}
loss_fn = RiskWeightedClinicalLoss(dosage_tokens=dosage_ids, penalty_multiplier=4.0)

logits = torch.randn(2, 4, 1200, requires_grad=True)
targets = torch.tensor([[10, 42, 5, 108], [999, 12, -100, 3]], dtype=torch.long)

loss = loss_fn(logits, targets)
loss.backward()

print(f"Calculated custom risk loss: {loss.item():.4f}")
print(f"Gradients computed successfully: {logits.grad is not None}")