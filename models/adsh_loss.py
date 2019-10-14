import torch.nn as nn


class ADSH_Loss(nn.Module):
    """
    Loss function of ADSH

    Args:
        code_length(int): Hashing code length.
        gamma(float): Hyper-parameter.
    """
    def __init__(self, code_length, gamma):
        super(ADSH_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma

    def forward(self, F, B, S, omega):
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum()
        quantization_loss = ((F - B[omega, :]) ** 2).sum()
        loss = (hash_loss + self.gamma * quantization_loss) / (F.shape[0] * B.shape[0])

        return loss
