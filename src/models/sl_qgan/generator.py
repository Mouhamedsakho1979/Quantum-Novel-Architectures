import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class QuantumGenerator(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Définition du circuit générateur
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # 1. Encodage du bruit latent (inputs)
            # On utilise RY pour mapper le bruit [-1, 1] vers des états quantiques
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # 2. Couches Variationnelles (Le "Cerveau" du générateur)
            for layer in range(n_layers):
                # Entanglement (Intrication) pour créer des corrélations complexes
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i+1])
                
                # Rotations ajustables
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
            
            # 3. Mesures : On retourne des valeurs entre -1 et 1 (PauliZ)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, noise):
        # Le bruit doit avoir la même taille que le nombre de qubits
        return self.q_layer(noise)