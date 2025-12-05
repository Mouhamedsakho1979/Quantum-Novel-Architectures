import pennylane as qml

def get_device(n_qubits):
    """Initialise le simulateur quantique."""
    return qml.device("default.qubit", wires=n_qubits)

def quantum_cascade_circuit(inputs, weights):
    """
    Le circuit logique du QFCC.
    - inputs : Données d'entrée (Features)
    - weights : Paramètres apprenables (Rotations)
    """
    n_qubits = len(inputs)
    n_layers = weights.shape[0]

    # 1. Encodage : On transforme les données classiques en états quantiques
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # 2. Cascade Variationnelle
    for layer in range(n_layers):
        # Entanglement (Intrication) : Crée les corrélations entre features
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Boucle fermée pour maximiser la connectivité
        qml.CNOT(wires=[n_qubits - 1, 0])
        
        # Rotations apprenables : C'est ici que le réseau "apprend"
        for i in range(n_qubits):
            qml.Rot(*weights[layer, i], wires=i)
            
    # 3. Mesure : On extrait l'information sur le premier qubit (Z-expectation)
    return qml.expval(qml.PauliZ(0))