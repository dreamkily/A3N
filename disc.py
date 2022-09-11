import torch
import torch.nn as nn

def discLoss(augmented_feature, target, margin=1):
    loss = 0
    counter = 0
    bsz = augmented_feature.size(0)
    mag = (augmented_feature ** 2).sum(1).expand(bsz, bsz)
    sim = augmented_feature.mm(augmented_feature.transpose(0, 1)) 
    dist = (mag + mag.transpose(0, 1) - 2 * sim)
    dist = torch.nn.functional.relu(dist).sqrt()
    for i in range(bsz):
        t_i = target[i].item()
        for j in range(i + 1, bsz):
            t_j = target[j].item()
            if t_i == t_j:
                l_ni = (margin - dist[i][target != t_i]).exp().sum()
                l_nj = (margin - dist[j][target != t_j]).exp().sum()
                l_n  = (l_ni + l_nj).log()
                l_p  = dist[i,j]
                loss += torch.nn.functional.relu(l_n + l_p) ** 2
                counter += 1
                
    return loss / (2 * counter)
