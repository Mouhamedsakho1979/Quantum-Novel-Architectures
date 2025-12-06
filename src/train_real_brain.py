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

# --- CONFIGURATION MULTI-MALADIES ---
Entrez.email = "researcher.senegal@quantum-lab.sn"
GENE_ID = "NM_000518" 
SEQ_LEN = 8
N_QUBITS = 6         # On garde 6 Qubits (Architecture App V3)
EPOCHS = 200         # On garde l'entraînement long
SAMPLES = 2000       # AUGMENTATION : Il faut plus de données pour apprendre 2 maladies
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
    
    print(f"🧬 Génération de {n_samples} échantillons (Drépanocytose + Cancer)...")
    
    count_hbb = 0
    count_cancer = 0
    
    for _ in range(n_samples):
        is_sick = np.random.rand() > 0.5
        start = np.random.randint(0, max_start)
        seq_str = list(real_gene[start : start + SEQ_LEN])
        
        label = 0.0
        if is_sick:
            label = 1.0
            mid = 3 # On cible le milieu pour l'apprentissage
            
            # --- LA CLÉ DU MULTI-CIBLES ---
            # Une fois sur deux, on injecte la Drépanocytose, l'autre fois le Cancer
            if np.random.rand() > 0.5:
                # DRÉPANOCYTOSE (HBB) -> GTG
                seq_str[mid] = 'G'
                seq_str[mid+1] = 'T'
                seq_str[mid+2] = 'G'
                count_hbb += 1
            else:
                # CANCER (Mutation Synthétique) -> GGG
                seq_str[mid] = 'G'
                seq_str[mid+1] = 'G'
                seq_str[mid+2] = 'G'
                count_cancer += 1
        
        # Encodage Numérique (0,1,2,3)
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        vec = [mapping.get(base, 0) for base in seq_str]
        X.append(vec)
        y.append(label)
    
    print(f"   - Cas Drépanocytose générés : {count_hbb}")
    print(f"   - Cas Cancer générés : {count_cancer}")
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# --- 2. ARCHITECTURE (Identique à l'App V3 pour compatibilité) ---
class GeneticQuantumScanner(nn.Module):
    def __init__(self, n_qubits, seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(4, n_qubits)
        self.position_embedding = nn.Embedding(seq_len, n_qubits)
        self.q_transformer = QuantumTransformerBlock(n_qubits, seq_len)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        tokens = self.token_embedding(x)
        pos = self.position_embedding(positions)
        x = tokens + pos 
        x = self.q_transformer(x)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)
        return self.classifier(x)

# --- 3. ENTRAÎNEMENT ---
def train():
    gene = download_gene()
    X_train, y_train = create_dataset(gene, SAMPLES)
    
    model = GeneticQuantumScanner(N_QUBITS, SEQ_LEN)
    
    # Scheduler dynamique
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    criterion = nn.BCELoss()
    
    print("\n🚀 Entraînement Multi-Pathologies en cours...")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        predicted = (outputs > 0.5).float()
        acc = (predicted == y_train).sum() / y_train.shape[0]
        
        if epoch % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Époque {epoch}/{EPOCHS} | Loss: {loss.item():.4f} | Précision: {acc*100:.1f}% | LR: {lr:.4f}")

    print(f"\n🏆 Score Final : {acc*100:.1f}%")
    
    torch.save(model.state_dict(), SAVE_PATH)
    print("💾 Nouveau Cerveau 'Bi-Spécialiste' sauvegardé.")

if __name__ == "__main__":
    train()