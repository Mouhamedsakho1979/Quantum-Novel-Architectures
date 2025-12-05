# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# On importe tes inventions
from src.models.qw_attn.transformer import QuantumTransformerBlock
from src.models.qagd.optimizer import QAGD  # On essaiera d'utiliser ton optimiseur plus tard

# --- CONFIGURATION AVANCÉE ---
SEQ_LEN = 8        # Longueur de la séquence (ex: 8 bases d'ADN)
N_QUBITS = 4       # Dimension de l'espace latent
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.01

# --- 1. GÉNÉRATION DE DONNÉES SYNTHÉTIQUES COMPLEXES ---
# Tâche : Classifier des séquences.
# Classe 0 : Séquences aléatoires.
# Classe 1 : Séquences contenant un "motif caché" (ex: une répétition périodique).

# --- 1. GÉNÉRATION DE DONNÉES GÉNÉTIQUES (ADN) ---
def generate_genomic_data(n_samples=200):
    """
    Simule des séquences d'ADN.
    Mapping : A=0, C=1, G=2, T=3
    On normalise entre 0 et 1 pour le quantique (divisé par 3).
    """
    X = []
    y = []
    
    # Motif "Malade" caché : La séquence 'A-C-G' (0, 1, 2) apparaît
    motif = [0, 1, 2] 
    
    for _ in range(n_samples):
        # 1. On crée une séquence aléatoire de base (ADN Sain)
        # On génère des entiers 0,1,2,3 (A,C,G,T)
        raw_seq = np.random.randint(0, 4, size=SEQ_LEN)
        label = 0 # Sain par défaut
        
        # 2. Pour la moitié des données, on injecte la maladie (le motif)
        if np.random.rand() > 0.5:
            start_pos = np.random.randint(0, SEQ_LEN - 3)
            raw_seq[start_pos:start_pos+3] = motif
            label = 1 # Malade
            
        # 3. Normalisation pour le circuit quantique (0 à 1)
        # A(0)->0.0, C(1)->0.33, G(2)->0.66, T(3)->1.0
        normalized_seq = raw_seq / 3.0
        
        X.append(normalized_seq.reshape(-1, 1))
        y.append(label)
        
    return np.array(X), np.array(y)


# --- 2. L'ARCHITECTURE Q-SEQ (Quantum Sequence Classifier) ---
class QSeqClassifier(nn.Module):
    def __init__(self, n_qubits, seq_len):
        super().__init__()
        # Projection de l'entrée (1 feature) vers l'espace quantique (n_qubits)
        self.embedding = nn.Linear(1, n_qubits)
        
        # Ton Transformer Quantique (Milestone 4)
        self.q_transformer = QuantumTransformerBlock(n_qubits, seq_len)
        
        # Pooling : On écrase la séquence pour avoir un seul vecteur résumé
        self.pooling = nn.AdaptiveAvgPool1d(1) 
        
        # Tête de classification finale
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 2) # Sortie : Classe 0 ou 1
        )

    def forward(self, x):
        # x shape: [batch, seq_len, 1]
        x = self.embedding(x)                 # -> [batch, seq_len, n_qubits]
        x = self.q_transformer(x)             # -> Attention Quantique
        
        # On transpose pour le pooling : [batch, n_qubits, seq_len]
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)       # -> [batch, n_qubits]
        
        return self.classifier(x)

# --- 3. L'EXPÉRIENCE ---
def run_experiment():
    print(f"\n🧬 Démarrage du Projet Q-Seq (Quantum Sequence Classifier)")
    print("Objectif : Détecter des motifs cachés dans des séquences bruitées.")

    # Données
    X, y = generate_genomic_data(n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Conversion PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Modèle
    model = QSeqClassifier(n_qubits=N_QUBITS, seq_len=SEQ_LEN)
    
    # Optimiseur : On utilise Adam ici pour stabiliser le Transformer au début
    # (Q-AGD est puissant mais délicat à régler sur des architectures profondes au début)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    accuracies = []

    print("\n--- Entraînement du Q-Seq ---")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Calcul précision
        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == y_train).sum().item() / len(y_train)
        accuracies.append(acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc*100:.1f}%")

    # Validation
    print("\n--- Test Final ---")
    with torch.no_grad():
        test_out = model(X_test)
        _, test_pred = torch.max(test_out.data, 1)
        test_acc = (test_pred == y_test).sum().item() / len(y_test)
    
    print(f"✅ Précision sur données inconnues : {test_acc*100:.2f}%")
    
    # Affichage
    try:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Loss')
        plt.title('Apprentissage')
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, color='green', label='Accuracy')
        plt.title('Précision')
        plt.show()
    except:
        pass

if __name__ == "__main__":
    run_experiment()