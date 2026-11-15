"""Mac GPU (MPS) workarounds. D-FINE's integral layer calls F.linear with a 1-D weight, whose MPS backward fails
('mat2 must be a matrix'); an explicit weighted sum is mathematically identical."""
import torch
from transformers.models.d_fine import modeling_d_fine as _m
def _integral_forward(self, pred_corners, project):
    b, q, _ = pred_corners.shape
    p = torch.softmax(pred_corners.reshape(-1, self.max_num_bins + 1), dim=1)
    w = project.to(p.device).reshape(-1)
    return (p * w).sum(-1).reshape(-1, 4).reshape(b, q, -1)
_m.DFineIntegral.forward = _integral_forward
