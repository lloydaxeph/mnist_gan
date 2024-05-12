import torch.nn as nn
import torch.utils.data

from utils import create_labels


class DiscriminatorLogitsLoss(nn.Module):
    def __init__(self, smoothing: float = 0.2, device: torch.device = None):
        """
        Ascend discriminator's stocastic gradient
        """
        super().__init__()
        self.loss_true = nn.BCEWithLogitsLoss()
        self.loss_false = nn.BCEWithLogitsLoss()
        self.smoothing = smoothing
        self.device = device

        self.register_buffer('labels_true', create_labels(n=256, r1=1.0 - smoothing, r2=1.0, device=self.device), False)
        self.register_buffer('labels_false', create_labels(n=256, r1=0.0, r2=smoothing, device=self.device), False)

    def forward(self, logits_true: torch.Tensor, logits_false: torch.Tensor) \
            -> (nn.BCEWithLogitsLoss, nn.BCEWithLogitsLoss):
        if len(logits_true) > len(self.labels_true):
            self.register_buffer('labels_true', create_labels(n=len(logits_true), r1=1.0 - self.smoothing, r2=1.0,
                                                              device=self.device), False)
        if len(logits_false) > len(self.labels_false):
            self.register_buffer('labels_false', create_labels(n=len(logits_false), r1=0.0, r2=self.smoothing,
                                                               device=self.device), False)

        return (self.loss_true(logits_true, self.labels_true[:len(logits_true)]),
                self.loss_false(logits_false, self.labels_false[:len(logits_false)]))


class GeneratorLogitsLoss(nn.Module):
    def __init__(self, smoothing: float = 0.2, device: torch.device = None):
        """
        Descend Generator's stocastic gradient
        """
        super().__init__()
        self.loss_true = nn.BCEWithLogitsLoss()
        self.smoothing = smoothing
        self.device = device

        self.register_buffer('fake_labels', create_labels(n=256, r1=1.0 - smoothing, r2=1.0, device=self.device), False)

    def forward(self, logits: torch.Tensor) -> nn.BCEWithLogitsLoss:
        if len(logits) > len(self.fake_labels):
            self.register_buffer('fake_labels', create_labels(n=len(logits), r1=1.0 - self.smoothing, r2=1.0,
                                                              device=self.device), False)
        return self.loss_true(logits, self.fake_labels[:len(logits)])
