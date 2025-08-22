import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def pirads_to_gleason_weak(self, pirads, num_class=6):
        weak_label = torch.zeros(pirads.size(0), num_class, dtype=torch.float, device=pirads.device)

 
        mask = pirads == 0
        weak_label[mask, 0] = 1

        
        mask = pirads == 1
        weak_label[mask, 1] = 0.9
        weak_label[mask, 2] = 0.1

        
        mask = pirads == 2
        weak_label[mask, 1] = 0.8
        weak_label[mask, 2] = 0.2

        
        mask = pirads == 3
        weak_label[mask, 1] = 0.7
        weak_label[mask, 2] = 0.3

        
        mask = pirads == 4
        weak_label[mask, 1] = 0.4
        weak_label[mask, 2] = 0.4
        weak_label[mask, 3] = 0.15
        weak_label[mask, 4] = 0.05
        
        mask = pirads == 5
        weak_label[mask, 1] = 0.2
        weak_label[mask, 2] = 0.4
        weak_label[mask, 3] = 0.15
        weak_label[mask, 4] = 0.1
        weak_label[mask, 5] = 0.05

        return weak_label

        
        
    def get_weak_label(self, targets, pirads, num_classes=6):
        onehot = torch.zeros((targets.size(0), num_classes), dtype=torch.float, device=pirads.device)
        non_negative = targets != -1
        # convert to one-hot for strong label
        onehot[non_negative] = F.one_hot(targets[non_negative], num_classes=num_classes).float()

        
        # convert to one-hot for weak label
        negative = targets == -1
        weak_label = self.pirads_to_gleason_weak(pirads)

        onehot[negative] = weak_label[negative]

        
        return onehot



    def forward(self, inputs, targets, pirads):
        """
        inputs: [N, C] logits
        targets: [N] class indices
        """
        

        weak_label = self.get_weak_label(targets, pirads)
        


        inputs = torch.softmax(inputs, dim=-1)

        pt = (inputs * weak_label).sum(dim=-1)
        # class_weight = self.alpha[targets].unsqueeze(-1)
        focal_loss = - torch.pow((1-pt), self.gamma) * torch.log(pt)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SimpleContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(SimpleContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, pos, neg):
        # Normalize embeddings
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)

        # Cosine similarity
        sim = F.cosine_similarity(pos, neg, dim=-1)

        # We want pos and neg to be different, so push similarity below margin
        loss = torch.clamp(sim - self.margin, min=0)
        return loss.mean()


if __name__ == '__main__':    
    x = torch.tensor([[10., 10., 10., 0, 0., 0.],
                      [10., 10., 10., 0, 0., 0.],
                      [10., 10., 10., 0, 0., 0.]])
    gleason = torch.tensor([2, -1, 3])
    pirads = torch.tensor([1, 4, 1])

    loss_function  = FocalLoss()

    print(loss_function(x, gleason, pirads))