# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- ASTUCE IMPORTANTE ---
# Cela permet à Python de trouver ton dossier 'src' peu importe où tu lances le script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Importation de ton modèle QFCC depuis les fichiers que tu viens de remplir
from src.models.qfcc.model import QFCC_Model

# =================CONFIGURATION=================
N_QUBITS = 4
N_LAYERS = 3
STEPS = 50
LR = 0.1
BATCH_SIZE = 16
SAVE_PATH = os.path.join(os.path.dirname(__file__), '../../results/qfcc_prototype.pth')

def run_experiment():
    print(f"\n🚀 Démarrage du PROTOCOLE QFCC (Milestone 1)")
    print(f"Architecture : {N_QUBITS} Qubits | {N_LAYERS} Couches Cascade")

    # 1. Préparation des Données (Simulation Small Data)
    # make_moons crée deux demi-cercles entrelacés (difficile pour un classifieur linéaire)
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Padding si on a moins de features que de qubits
    if X_scaled.shape[1] < N_QUBITS:
        padding = np.zeros((X_scaled.shape[0], N_QUBITS - X_scaled.shape[1]))
        X_processed = np.hstack((X_scaled, padding))
    else:
        X_processed = X_scaled[:, :N_QUBITS]

    # Tensors PyTorch
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2)
    X_train = torch.tensor(X_train, requires_grad=False).float()
    y_train = torch.tensor(y_train, requires_grad=False).long()
    
    # 2. Initialisation du Modèle
    model = QFCC_Model(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 3. Boucle d'Entraînement
    loss_history = []
    print("\n--- Entraînement en cours... ---")
    
    for step in range(STEPS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if step % 10 == 0:
            _, preds = torch.max(outputs, 1)
            acc = (preds == y_train).sum().item() / len(y_train)
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | Train Accuracy: {acc*100:.1f}%")

    # 4. Sauvegarde et Résultats
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Modèle sauvegardé dans : {SAVE_PATH}")
    
    # Affichage graphique
    try:
        plt.plot(loss_history)
        plt.title("Convergence du QFCC")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()
    except:
        print("Graphique non affiché, mais calcul terminé.")

if __name__ == "__main__":
    run_experiment()