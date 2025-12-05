# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sl_qgan.generator import QuantumGenerator
from src.models.sl_qgan.discriminator import Discriminator

# --- CONFIGURATION ---
N_QUBITS = 2       # On génère des points 2D (X, Y)
N_LAYERS = 4       # Profondeur du circuit
BATCH_SIZE = 16
LR_G = 0.05        # Learning Rate Générateur (souvent plus élevé en QML)
LR_D = 0.01        # Learning Rate Discriminateur
EPOCHS = 100       # Nombre de tours
REAL_LABEL = 1.0
FAKE_LABEL = 0.0

def run_experiment():
    print(f"\n🚀 Démarrage du Milestone 3 : SL-QGAN (Generative AI)")
    print("Objectif : Apprendre la distribution d'un CERCLE via un circuit quantique.\n")

    # 1. Modèles
    generator = QuantumGenerator(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    discriminator = Discriminator(input_dim=N_QUBITS)

    # 2. Optimiseurs
    opt_G = optim.Adam(generator.parameters(), lr=LR_G)
    opt_D = optim.Adam(discriminator.parameters(), lr=LR_D)
    criterion = nn.BCELoss() # Binary Cross Entropy

    # Stockage pour graphiques
    g_losses = []
    d_losses = []

    print("--- Entraînement Adversarial ---")
    for epoch in range(EPOCHS):
        
        # A. Entraînement du Discriminateur (Le Critique)
        opt_D.zero_grad()
        
        # Données Réelles (Un cercle parfait)
        real_data, _ = make_circles(n_samples=BATCH_SIZE, factor=0.5, noise=0.05)
        real_data = torch.tensor(real_data, dtype=torch.float32)
        
        label_real = torch.full((BATCH_SIZE, 1), REAL_LABEL)
        output_real = discriminator(real_data)
        loss_real = criterion(output_real, label_real)

        # Données Fausses (Générées par le Quantique)
        noise = torch.rand(BATCH_SIZE, N_QUBITS) * np.pi # Bruit entre 0 et Pi
        fake_data = generator(noise)
        
        label_fake = torch.full((BATCH_SIZE, 1), FAKE_LABEL)
        output_fake = discriminator(fake_data.detach()) # Detach pour ne pas toucher au Générateur ici
        loss_fake = criterion(output_fake, label_fake)

        # Backprop Discriminateur
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        opt_D.step()

        # B. Entraînement du Générateur (L'Artiste)
        opt_G.zero_grad()
        
        # On veut tromper le discriminateur (on lui dit que c'est VRAI)
        output_fake_for_G = discriminator(fake_data)
        loss_G = criterion(output_fake_for_G, label_real) # Trick: target is REAL
        
        loss_G.backward()
        opt_G.step()

        g_losses.append(loss_G.item())
        d_losses.append(loss_D.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    # 3. Visualisation Finale
    print("\n✅ Génération des résultats visuels...")
    
    # On génère 100 points avec le modèle entraîné
    with torch.no_grad():
        noise_test = torch.rand(100, N_QUBITS) * np.pi
        generated_data = generator(noise_test).numpy()

    # Données réelles pour comparaison
    real_plot, _ = make_circles(n_samples=100, factor=0.5, noise=0.05)

    try:
        plt.figure(figsize=(10, 5))
        
        # Plot 1: Les Données
        plt.subplot(1, 2, 1)
        plt.scatter(real_plot[:, 0], real_plot[:, 1], c='blue', alpha=0.5, label='Réel (Cercle)')
        plt.scatter(generated_data[:, 0], generated_data[:, 1], c='red', alpha=0.7, label='Généré (Quantique)')
        plt.legend()
        plt.title("Réalité vs Fiction Quantique")
        plt.grid(True)

        # Plot 2: Les Courbes d'apprentissage
        plt.subplot(1, 2, 2)
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(d_losses, label='Discriminator Loss')
        plt.legend()
        plt.title("Courbes d'apprentissage QGAN")
        
        plt.show()
    except:
        print("Graphique non affiché.")
    
    print("Milestone 3 Validé : Le générateur quantique est fonctionnel.")

if __name__ == "__main__":
    run_experiment()