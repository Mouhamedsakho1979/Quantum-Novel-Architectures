import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Un réseau de neurones classique simple mais efficace
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(0.2),  # LeakyReLU est standard pour les GANs
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()        # Sortie entre 0 (Faux) et 1 (Vrai)
        )

    def forward(self, x):
        return self.net(x)