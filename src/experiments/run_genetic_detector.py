# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.qw_attn.transformer import QuantumTransformerBlock

# --- ⚡️ CONFIGURATION OPTIMISÉE POUR LE SUCCÈS ⚡️ ---
SEQ_LEN = 8        # On raccourcit un peu pour faciliter l'apprentissage
N_QUBITS = 4       # Puissance
EPOCHS = 80        # Plus de temps pour apprendre (au lieu de 40)
ANOMALY_RATE = 0.3 # 30% de malades (pour qu'elle ait assez d'exemples)

# --- GÉNÉRATEUR ADN ---
def generate_dna_data(n_samples=500):
    X = []
    y = []
    mutation_pattern = [2, 2, 2] # Le motif "GGG" (Cancer virtuel)
    
    print(f"🔬 Génération de {n_samples} séquences ADN...")
    print(f"⚠️ Taux d'anomalie : {ANOMALY_RATE*100}%")
    
    for _ in range(n_samples):
        raw_seq = np.random.randint(0, 4, size=SEQ_LEN)
        label = 0 
        
        if np.random.rand() < ANOMALY_RATE:
            start = np.random.randint(0, SEQ_LEN - 3)
            raw_seq[start:start+3] = mutation_pattern
            label = 1 
            
        normalized_seq = raw_seq / 3.0
        X.append(normalized_seq.reshape(-1, 1))
        y.append(label)
        
    return np.array(X), np.array(y)

def decode_dna(seq_vector):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq_integers = (seq_vector * 3).round().astype(int).flatten()
    return "".join([mapping.get(x, '?') for x in seq_integers])

# --- CERVEAU QUANTIQUE ---
class GeneticQuantumScanner(nn.Module):
    def __init__(self, n_qubits, seq_len):
        super().__init__()
        self.embedding = nn.Linear(1, n_qubits)
        self.q_transformer = QuantumTransformerBlock(n_qubits, seq_len)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.q_transformer(x)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)
        return self.classifier(x)

# --- EXPÉRIENCE ---
def run_experiment():
    print(f"\n🧬 Démarrage du PROTOCOLE : Détection d'Anomalie Génétique")
    
    X, y = generate_dna_data(n_samples=400) # Un peu moins de données pour aller plus vite
    
    print(f"\nExemple Sain   : {decode_dna(X[0])}")
    if np.sum(y) > 0:
        malade_idx = np.where(y==1)[0][0]
        print(f"Exemple Malade : {decode_dna(X[malade_idx])} (Contient GGG)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # On force l'IA à s'intéresser aux malades (Poids x3)
    weights = torch.tensor([1.0, 3.0])
    
    model = GeneticQuantumScanner(n_qubits=N_QUBITS, seq_len=SEQ_LEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02) # Vitesse d'apprentissage doublée
    criterion = nn.CrossEntropyLoss(weight=weights)

    print("\n--- Analyse en cours... (Patience : ~2 min) ---")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            _, preds = torch.max(outputs, 1)
            acc = (preds == y_train).sum().item() / len(y_train)
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Précision Train: {acc*100:.1f}%")

    print("\n--- RÉSULTATS DU SCANNER ---")
    with torch.no_grad():
        test_out = model(X_test)
        _, test_pred = torch.max(test_out, 1)
        
        tp = ((test_pred == 1) & (y_test == 1)).sum().item()
        total_malades = (y_test == 1).sum().item()
        
        print(f"Malades dans le test : {total_malades}")
        print(f"Détectés par l'IA    : {tp}")
        
        if total_malades > 0:
            sensibilite = tp / total_malades
            print(f"✅ SCORE DE DÉTECTION : {sensibilite*100:.1f}%")
            
            if sensibilite > 0.8:
                print("🏆 SUCCÈS TOTAL : Ton prototype fonctionne !")
            elif sensibilite > 0.5:
                print("⚠️ PROMETTEUR : Elle commence à comprendre.")
            else:
                print("❌ ÉCHEC : Relance l'entraînement.")

if __name__ == "__main__":
    run_experiment()