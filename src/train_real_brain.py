# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
from Bio import Entrez, SeqIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.models.qw_attn.transformer import QuantumTransformerBlock

# --- CONFIGURATION ULTIME ---
Entrez.email = "researcher.senegal@quantum-lab.sn"
GENE_ID = "NM_000518" 
SEQ_LEN = 8
N_QUBITS = 6         # UPGRADE : 6 Qubits (Plus d'espace mental)
EPOCHS = 150
SAMPLES = 1000
SAVE_PATH = "src/q_seq_brain.pth"

# --- 1. DATASET ---
def download_gene():
    try:
        handle = Entrez.efetch(db="nucleotide", id=GENE_ID, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        return str(record.seq)
    except:
        return "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGT" * 20

def create_dataset(real_gene, n_samples):
    X, y = [], []
    max_start = len(real_gene) - SEQ_LEN - 1
    
    print(f"🧬 Génération de {n_samples} échantillons avec Positionnement...")
    
    for _ in range(n_samples):
        is_sick = np.random.rand() > 0.5
        start = np.random.randint(0, max_start)
        seq_str = list(real_gene[start : start + SEQ_LEN])
        
        label = 0.0
        if is_sick:
            # On place la mutation TOUJOURS au même endroit relatif pour aider l'IA au début
            # C'est comme lui apprendre à regarder au centre de l'image
            mid = 3 
            seq_str[mid] = 'G'
            seq_str[mid+1] = 'T'
            seq_str[mid+2] = 'G'
            label = 1.0
        
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        vec = [mapping.get(base, 0) for base in seq_str]
        X.append(vec)
        y.append(label)
        
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# --- 2. ARCHITECTURE AVEC "GPS" (Positional Encoding) ---
class GeneticQuantumScanner(nn.Module):
    def __init__(self, n_qubits, seq_len):
        super().__init__()
        
        # 1. Embedding des Lettres (A,C,G,T -> Vecteur)
        self.token_embedding = nn.Embedding(4, n_qubits)
        
        # 2. Embedding de Position (Le GPS : Position 1, 2, 3... -> Vecteur)
        # C'est ÇA qui manquait !
        self.position_embedding = nn.Embedding(seq_len, n_qubits)
        
        # 3. Cœur Quantique
        self.q_transformer = QuantumTransformerBlock(n_qubits, seq_len)
        
        # 4. Décision
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.1), # Évite le par cœur
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x est [Batch, SeqLen]
        batch_size, seq_len = x.shape
        
        # On crée les indices de position [0, 1, 2, 3, 4, 5, 6, 7]
        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1)
        
        # Fusion : Sens des Lettres + Sens de la Position
        tokens = self.token_embedding(x)
        pos = self.position_embedding(positions)
        
        x = tokens + pos # L'addition magique des Transformers
        
        # Analyse Quantique
        x = self.q_transformer(x)
        
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)
        return self.classifier(x)

# --- 3. ENTRAÎNEMENT ---
def train():
    gene = download_gene()
    X_train, y_train = create_dataset(gene, SAMPLES)
    
    model = GeneticQuantumScanner(N_QUBITS, SEQ_LEN)
    
    # Scheduler : On commence vite, on ralentit à la fin
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    criterion = nn.BCELoss()
    
    print("\n🚀 Lancement de l'Entraînement ULTIME...")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step() # On réduit la vitesse
        
        predicted = (outputs > 0.5).float()
        acc = (predicted == y_train).sum() / y_train.shape[0]
        
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Époque {epoch}/{EPOCHS} | Loss: {loss.item():.4f} | Précision: {acc*100:.1f}% | LR: {current_lr:.4f}")

    print(f"\n🏆 Score Final : {acc*100:.1f}%")
    
    if acc > 0.90:
        print("✅ VICTOIRE TOTALE : L'IA est opérationnelle.")
    else:
        print(f"⚠️ Score : {acc*100:.1f}%. C'est suffisant pour une démo, mais retente si tu veux.")
        
    torch.save(model.state_dict(), SAVE_PATH)
    print("💾 Cerveau sauvegardé.")

if __name__ == "__main__":
    train()