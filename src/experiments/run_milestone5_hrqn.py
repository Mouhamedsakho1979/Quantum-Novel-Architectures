# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.hrqn.network import HRQN_Model

# --- CONFIGURATION ---
N_QUBITS = 2       # Petit système compact
N_LAYERS = 6       # Profondeur de l'évolution
STEPS = 150
LR = 0.05

def target_function(x):
    """Une fonction non-linéaire complexe : x * cos(3*x)"""
    return x * np.cos(3 * x)

def run_experiment():
    print(f"\n🚀 Démarrage du Milestone 5 : HRQN (Hamiltonian Network)")
    print("Objectif : Apprendre la dynamique d'une fonction physique complexe.")

    # 1. Préparation des Données
    # On génère des points entre -1 et 1
    x_numpy = np.linspace(-1, 1, 50).reshape(-1, 1)
    y_numpy = target_function(x_numpy)
    
    # Pour l'encodage sur 2 qubits, on duplique l'entrée (redundancy)
    X_input = np.hstack([x_numpy, x_numpy]) 
    
    X_train = torch.tensor(X_input, dtype=torch.float32)
    y_train = torch.tensor(y_numpy, dtype=torch.float32)

    # 2. Modèle
    model = HRQN_Model(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    loss_history = []
    
    print("\n--- Évolution du Système (Entraînement) ---")
    for step in range(STEPS):
        optimizer.zero_grad()
        
        preds = model(X_train)
        loss = criterion(preds, y_train)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if step % 20 == 0:
            print(f"Step {step:03d} | Energy Loss: {loss.item():.5f}")

    # 3. Validation Visuelle
    print("\n✅ Affichage de la physique apprise...")
    with torch.no_grad():
        y_pred = model(X_train).numpy()

    try:
        plt.figure(figsize=(8, 5))
        plt.plot(x_numpy, y_numpy, label='Réalité (Physique)', color='black', linewidth=2)
        plt.plot(x_numpy, y_pred, label='Modèle HRQN (Quantique)', color='purple', linestyle='--', marker='o')
        plt.title("HRQN : Apprentissage par Hamiltonien")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except:
        pass
        
    print("\n🎉 FÉLICITATIONS ! Tu as implémenté les 5 Architectures.")
    print("Ton dossier 'Quantum_Novel_Architectures' est complet.")

if __name__ == "__main__":
    run_experiment()