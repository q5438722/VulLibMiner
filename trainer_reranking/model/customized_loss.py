import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon=0, pos_weight=None, **kwargs):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.pos_weight = pos_weight

    def forward(self, scores, labels):
        if self.pos_weight:
            loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]).to(scores.device))
        else:
            loss_func = nn.BCEWithLogitsLoss()
        target_smooth = (1 - self.epsilon) * labels + self.epsilon / 2
        return loss_func(scores, target_smooth)


class FocalLoss(nn.Module):
    def __init__(self, alpha=-1.0, gamma=2, **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, scores, targets):
        p = torch.sigmoid(scores)
        ce_loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean() * 1000



#### https://github.com/CoinCheung/gdGPT/blob/master/models/__init__.py
class LMCrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ignore_index=-100)

    def forward(self, lm_logits, labels):
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_length, vocab_size = shift_logits.shape
        return super().forward(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length)
        )


