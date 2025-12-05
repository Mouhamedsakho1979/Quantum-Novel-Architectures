import pennylane as qml

def hamiltonian_evolution_layer(weights, wires, delta_t=0.1):
    """
    Simule une évolution temporelle sous un Hamiltonien H.
    U = exp(-i * H * t)
    
    L'idée est que le réseau apprend les coefficients de l'Hamiltonien (l'énergie)
    pour guider les qubits vers la bonne réponse de manière fluide.
    """
    n_qubits = len(wires)
    
    # Trotterization (Approximation de l'évolution continue)
    # H = Somme(Rotations X) + Somme(Interactions ZZ)
    
    # 1. Terme d'interaction (Entanglement naturel)
    # C'est la "force" qui relie les qubits entre eux
    for i in range(n_qubits - 1):
        qml.IsingZZ(weights[i, 0] * delta_t, wires=[wires[i], wires[i+1]])
    # Fermeture de la boucle
    if n_qubits > 1:
        qml.IsingZZ(weights[n_qubits-1, 0] * delta_t, wires=[wires[n_qubits-1], wires[0]])
        
    # 2. Terme transversal (Champ externe)
    # C'est le réglage individuel de chaque qubit
    for i in range(n_qubits):
        qml.RX(weights[i, 1] * delta_t, wires=wires[i])
        qml.RZ(weights[i, 2] * delta_t, wires=wires[i])