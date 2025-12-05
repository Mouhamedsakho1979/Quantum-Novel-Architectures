# -*- coding: utf-8 -*-
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Ajout du chemin pour trouver les modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.qagd.optimizer import QAGD

def run_experiment():
    print("\n🚀 Démarrage du Milestone 2 : Q-AGD Optimizer")
    print("Objectif : Minimiser une fonction non-convexe SANS calcul de gradient classique.\n")

    # --- 1. Définition du problème ---
    # On utilise la fonction de Rosenbrock (célèbre test difficile pour les optimiseurs)
    # f(x, y) = (a - x)^2 + b * (y - x^2)^2
    # Minimum global en (a, a^2). Avec a=1, b=100 -> min en (1, 1) = 0.
    
    def rosenbrock(params):
        x, y = params[0], params[1]
        return (1 - x)**2 + 100 * (y - x**2)**2

    # Point de départ aléatoire (loin de la solution 1,1)
    start_point = torch.tensor([-1.5, -1.0], requires_grad=False) # Pas de gradient nécessaire !
    params_qagd = start_point.clone()
    
    # --- 2. Configuration de l'optimiseur Q-AGD ---
    # Note : Pas besoin de "requires_grad=True" car Q-AGD estime le gradient lui-même
    optimizer = QAGD([params_qagd], lr=0.5, perturbation=0.1)

    history = []
    
    print(f"Point de départ : {params_qagd.numpy()}")
    print("Lancement de l'optimisation...")

    # --- 3. Boucle d'optimisation ---
    for i in range(100):
        # La closure est la fonction qui permet à l'optimiseur de réévaluer la loss
        def closure():
            return rosenbrock(params_qagd)
        
        loss = optimizer.step(closure)
        history.append(loss.item())

        if i % 10 == 0:
            print(f"Iter {i:03d} | Loss: {loss.item():.6f} | Position: {params_qagd.numpy()}")

    # --- 4. Résultats ---
    print(f"\n✅ Terminé. Position finale : {params_qagd.numpy()}")
    print(f"Cible théorique : [1.  1.]")
    
    # Graphique
    try:
        plt.plot(history, label='Q-AGD Loss')
        plt.yscale('log') # Échelle logarithmique pour mieux voir la convergence
        plt.title("Convergence de l'optimiseur Q-AGD (Rosenbrock)")
        plt.xlabel("Itérations")
        plt.ylabel("Loss (Log Scale)")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()
    except:
        pass

if __name__ == "__main__":
    run_experiment()