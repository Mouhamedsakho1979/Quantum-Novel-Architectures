import torch
import torch.nn as nn
from .attention import QuantumAttention

class QuantumTransformerBlock(nn.Module):
    def __init__(self, n_qubits, seq_len):
        super().__init__()
        
        # 1. Attention Quantique
        self.attn = QuantumAttention(n_qubits, seq_len)
        self.norm1 = nn.LayerNorm(n_qubits)
        
        # 2. Feed Forward (Réseau dense classique pour digérer l'info)
        self.ff = nn.Sequential(
            nn.Linear(n_qubits, n_qubits * 2),
            nn.ReLU(),
            nn.Linear(n_qubits * 2, n_qubits)
        )
        self.norm2 = nn.LayerNorm(n_qubits)

    def forward(self, x):
        # Connexion résiduelle 1 (Add & Norm)
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        
        # Connexion résiduelle 2 (Add & Norm)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class QuantumSequenceModel(nn.Module):
    def __init__(self, n_qubits, seq_len, output_dim):
        super().__init__()
        self.embedding = nn.Linear(1, n_qubits) # Entrée simple (1 chiffre) vers Dimension Quantique
        self.transformer = QuantumTransformerBlock(n_qubits, seq_len)
        self.head = nn.Linear(n_qubits * seq_len, output_dim) # Prédiction finale

    def forward(self, x):
        # x shape: [batch, seq_len, 1]
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.view(x.shape[0], -1) # Flatten
        return self.head(x)