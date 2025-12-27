import torch

def compute_risk(mask, danger_class=1):
    """
    Computes fraction of dangerous pixels in a segmentation mask.

    mask: Tensor [H, W] or [B, H, W]
    """
    if mask.dim() == 2:
        total_pixels = mask.numel()
        danger_pixels = (mask == danger_class).sum().item()
        return danger_pixels / total_pixels

    elif mask.dim() == 3:
        risks = []
        for m in mask:
            total_pixels = m.numel()
            danger_pixels = (m == danger_class).sum().item()
            risks.append(danger_pixels / total_pixels)
        return risks


def safety_decision(risk, threshold=0.05):
    """
    Converts risk score into SAFE / NOT SAFE decision.
    """
    return "SAFE" if risk < threshold else "NOT SAFE"
