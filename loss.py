import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.

    Courtesy of https://github.com/maticvl/dataHacker/blob/master/pyTorch/014_siameseNetwork.ipynb
    as this is where this loss function comes from. Although, this loss function is literally just
    the implementation of the Contrastive Loss formula.
    """
    def __init__(self, margin=2.0) -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_1, output_2, label):
        dist = F.pairwise_distance(output_1, output_2, keepdim=True)

        loss = torch.mean(
            (1 - label) * torch.pow(dist, 2) + (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        
        return loss
