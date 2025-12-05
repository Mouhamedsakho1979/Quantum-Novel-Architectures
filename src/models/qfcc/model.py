import torch
import torch.nn as nn
import pennylane as qml
from .circuit import get_device, quantum_cascade_circuit

class QFCC_Model(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Initialisation du device quantique
        self.dev = get_device(n_qubits)
        
        # Création du QNode (le nœud quantique connectable à PyTorch)
        self.qnode = qml.QNode(quantum_cascade_circuit, self.dev, interface="torch")
        
        # Définition de la forme des poids : [Couches, Qubits, 3 Angles]
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Couche Quantique Hybride
        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        
        # Couche Classique finale (Classification binaire)
        # Elle prend la sortie du quantique et décide de la classe
        self.classical_head = nn.Linear(1, 2) 

    def forward(self, x):
        # Passage dans le circuit quantique
        q_out = self.q_layer(x)
        
        # Remise en forme pour le réseau classique
        q_out = q_out.view(-1, 1)
        
        # Classification finale
        return self.classical_head(q_out)