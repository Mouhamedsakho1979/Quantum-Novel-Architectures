# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.qw_attn.transformer import QuantumSequenceModel

# --- CONFIGURATION ---
SEQ_LEN = 5        # On regarde 5 points passés
N_QUBITS = 4       # Dimension du modèle (d_model)
STEPS = 100
LR = 0.01

def create_dataset(steps=200):
    """Crée une onde sinusoïdale pour la prédiction."""
    t = np.linspace(0, 4*np.pi, steps)
    data = np.sin(t)
    X, y = [], []
    for i in range(len(data) - SEQ_LEN):
        X.append(data[i:i+SEQ_LEN])
        y.append(data[i+SEQ_LEN]) # On prédit le point suivant
    return np.array(X), np.array(y)

def run_experiment():
    print(f"\n🚀 Démarrage du Milestone 4 : Quantum Transformer")
    print("Objectif : Prédiction de séries temporelles (Sine Wave).")

    # 1. Données
    X, y = create_dataset()
    # Format PyTorch : [Batch, Seq_Len, Feature_Dim]
    X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) 
    y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    
    # 2. Modèle
    model = QuantumSequenceModel(n_qubits=N_QUBITS, seq_len=SEQ_LEN, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # Mean Squared Error (Regression)

    losses = []
    
    print("\n--- Entraînement du Transformer ---")
    for step in range(STEPS):
        optimizer.zero_grad()
        
        preds = model(X_train)
        loss = criterion(preds, y_train)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"Step {step:03d} | MSE Loss: {loss.item():.5f}")

    # 3. Validation Visuelle
    print("\n✅ Test de prédiction...")
    with torch.no_grad():
        predicted = model(X_train).numpy()

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(y, label='Vraie Séquence (Sinus)', color='black', linestyle='--')
        plt.plot(predicted, label='Prédiction Q-Transformer', color='red')
        plt.title("Quantum Transformer : Prédiction de Séquence")
        plt.legend()
        plt.show()
    except:
        pass
    
    print("Milestone 4 Terminé.")

if __name__ == "__main__":
    run_experiment()