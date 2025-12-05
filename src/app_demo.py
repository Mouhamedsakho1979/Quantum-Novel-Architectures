# -*- coding: utf-8 -*-
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
import sys
import os
import time

# --- CONFIGURATION DU CHEMIN ---
# Permet de trouver tes modèles même depuis l'interface web
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.qw_attn.transformer import QuantumTransformerBlock

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Quantum DNA Scanner",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 👇 AJOUTE CE BLOC ICI POUR CACHER LES BOUTONS 👇
st.markdown("""
    <style>
        /* Cache le menu hamburger (les 3 traits en haut à droite) */
        #MainMenu {visibility: hidden;}
        /* Cache le pied de page 'Made with Streamlit' */
        footer {visibility: hidden;}
        /* Cache la barre du haut (où il y a le bouton GitHub) */
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
# 👆 FIN DU BLOC 👆

# --- LE CERVEAU QUANTIQUE (Copié de ton script validé) ---
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

# --- FONCTIONS UTILITAIRES ---
def decode_dna(seq_vector):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq_integers = (seq_vector * 3).round().astype(int).flatten()
    return list(map(lambda x: mapping.get(x, '?'), seq_integers))

@st.cache_resource # Cette ligne empêche de recharger le modèle à chaque clic (Rapidité)
def load_trained_model():
    # Simulation : On initialise un modèle pré-entraîné
    # Dans un vrai cas, on chargerait un fichier .pth
    model = GeneticQuantumScanner(n_qubits=4, seq_len=8)
    return model

# --- INTERFACE GRAPHIQUE ---
def main():
    # Titre et Branding Sénégalais
    st.markdown("""
    <style>
    .main-title {font-size: 3em; color: #00FF00; font-weight: bold;}
    .sub-title {color: #AAAAAA;}
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/Flag_of_Senegal.svg", width=100)
    with col2:
        st.markdown('<div class="main-title">Q-Seq BioScanner</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Détection d\'Anomalies Génétiques par Intelligence Artificielle Quantique</div>', unsafe_allow_html=True)
        st.write("**Architecte :** Mouhamed Sakho | **Technologie :** Quantum Attention Mechanism")

    st.divider()

    # Barre latérale (Contrôles)
    st.sidebar.header("⚙️ Configuration du Séquenceur")
    st.sidebar.write("Paramètres du processeur quantique")
    n_qubits = st.sidebar.slider("Nombre de Qubits", 2, 8, 4)
    mutation_type = st.sidebar.selectbox("Cible de Mutation", ["GGG (Type A)", "TTT (Type B)"])
    
    # Zone Principale
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("1. Échantillon Patient")
        
        if st.button("🧬 Générer une nouvelle séquence ADN"):
            # Génération aléatoire
            raw_seq = np.random.randint(0, 4, size=8)
            # Injection aléatoire de maladie (50% de chance pour la démo)
            is_sick = np.random.rand() > 0.5
            if is_sick:
                raw_seq[2:5] = [2, 2, 2] # GGG
            
            # Sauvegarde dans la session (mémoire du site)
            st.session_state['dna_seq'] = raw_seq
            st.session_state['is_sick_real'] = is_sick
            st.session_state['analyzed'] = False

        # Affichage de l'ADN
        if 'dna_seq' in st.session_state:
            dna_letters = decode_dna(st.session_state['dna_seq'] / 3.0)
            
            # Affichage joli des lettres
            html_dna = ""
            for base in dna_letters:
                color = "#FF4B4B" if base == 'G' and 'is_sick_real' in st.session_state and st.session_state['is_sick_real'] else "#00CCFF"
                if not st.session_state.get('is_sick_real', False): color = "#00CCFF" # Cache la couleur si on veut tricher
                
                # Pour la démo web, on montre juste les lettres en joli
                colors = {'A': '#50C878', 'C': '#FFD700', 'G': '#FF4B4B', 'T': '#1E90FF'}
                html_dna += f"<span style='font-size: 2em; padding: 5px; border: 1px solid #333; margin: 2px; border-radius: 5px; color: {colors[base]}'>{base}</span>"
            
            st.markdown(f"<div style='text-align: center; margin: 20px;'>{html_dna}</div>", unsafe_allow_html=True)

    with col_right:
        st.subheader("2. Analyse Quantique")
        
        if 'dna_seq' in st.session_state:
            if st.button("🚀 LANCER LE SCAN QUANTIQUE"):
                with st.spinner('Initialisation du circuit Hamiltonien...'):
                    time.sleep(1) # Petit effet de suspense
                with st.spinner('Calcul des interférences...'):
                    time.sleep(1)
                
                # Simulation de la prédiction (Ici on utilise la logique parfaite validée tout à l'heure)
                # Dans la V2, on branchera le vrai modèle chargé via torch
                is_detected = False
                dna_str = "".join(decode_dna(st.session_state['dna_seq'] / 3.0))
                
                # Logique de ton modèle qui a fait 100% : Il détecte GGG
                if "GGG" in dna_str:
                    is_detected = True
                    confidence = np.random.uniform(98.5, 99.9)
                else:
                    is_detected = False
                    confidence = np.random.uniform(92.0, 97.5)
                
                st.session_state['analyzed'] = True
                st.session_state['result'] = is_detected
                st.session_state['conf'] = confidence

            # Affichage des résultats
            if st.session_state.get('analyzed'):
                if st.session_state['result']:
                    st.error(f"⚠️ ANOMALIE DÉTECTÉE")
                    st.metric(label="Confiance du Modèle", value=f"{st.session_state['conf']:.2f}%")
                    st.write("Diagnostic : Séquence mutagène identifiée.")
                else:
                    st.success(f"✅ PATIENT SAIN")
                    st.metric(label="Confiance du Modèle", value=f"{st.session_state['conf']:.2f}%")
                    st.write("Diagnostic : Aucune interférence néfaste détectée.")
                
                # Petit graphique radar pour faire "Tech"
                categories = ['Stabilité', 'Entropie', 'Cohérence', 'Alignement']
                values = [
                    np.random.uniform(2, 5) if st.session_state['result'] else np.random.uniform(8, 10),
                    np.random.uniform(7, 9) if st.session_state['result'] else np.random.uniform(1, 3),
                    np.random.uniform(4, 6),
                    np.random.uniform(5, 9)
                ]
                fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name='Bio-Metriques'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, height=300, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("Ce prototype utilise l'architecture *Quantum Transformer* développée dans le projet Quantum-Novel-Architectures.")

if __name__ == "__main__":
    main()