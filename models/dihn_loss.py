import torch
import torch.nn as nn


class DIHN_Loss(nn.Module):
    """
    Loss function of DIHN.

    Args:
        code_length(int): Hashing code length.
        gamma, mu(float): Hyper-parameter.
    """
    def __init__(self, code_length, gamma, mu):
        super(DIHN_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma
        self.mu = mu

    def forward(self, F, B, S, omega):
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum()
        quantization_loss = ((F - B[omega, :]) ** 2).sum()
        correlation_loss = (F @ torch.ones(F.shape[1], 1, device=F.device)).sum()
        loss = (hash_loss + self.gamma * quantization_loss + self.mu * correlation_loss) / (F.shape[0] * B.shape[0])

        return loss
