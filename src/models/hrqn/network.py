import torch
import torch.nn as nn
import pennylane as qml
from .dynamics import hamiltonian_evolution_layer

class HRQN_Model(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # 1. Encodage initial (Etat du système au temps t=0)
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            
            # 2. Évolution Temporelle par couches (ResNet-like)
            # Chaque couche est un petit pas de temps dt
            for layer_idx in range(n_layers):
                layer_weights = weights[layer_idx]
                hamiltonian_evolution_layer(layer_weights, wires=range(n_qubits))
            
            # 3. Mesure de l'énergie finale (PauliZ sur le premier qubit)
            return qml.expval(qml.PauliZ(0))

        # Poids : [Layers, Qubits, 3 coeffs (Interaction, X-field, Z-field)]
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # Scaling final pour adapter la sortie à la plage des données réelles
        self.final_scale = nn.Linear(1, 1)

    def forward(self, x):
        # x shape: [batch, n_qubits]
        q_out = self.q_layer(x)
        return self.final_scale(q_out.view(-1, 1))