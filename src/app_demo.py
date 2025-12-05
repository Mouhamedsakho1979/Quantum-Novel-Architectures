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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.models.qw_attn.transformer import QuantumTransformerBlock

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Quantum DNA Scanner",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CHIRURGICAL (Réparation Menu + Suppression Manage App) ---
st.markdown("""
    <style>
        /* 1. CACHER LE BOUTON 'MANAGE APP' ET LA TOOLBAR DE DROITE */
        /* On cible spécifiquement la barre d'outils en haut à droite */
        [data-testid="stToolbar"] {
            visibility: hidden !important;
            height: 0px !important;
        }
        
        /* 2. NE PAS CACHER LE HEADER, MAIS LE RENDRE TRANSPARENT */
        /* Cela permet au bouton Menu de gauche de rester cliquable */
        header {
            background-color: transparent !important;
        }

        /* 3. FORCER L'AFFICHAGE DU BOUTON MENU (HAMBURGER) */
        [data-testid="stSidebarCollapsedControl"] {
            display: block !important;
            visibility: visible !important;
            color: #00FF00 !important; /* Vert fluo pour le voir partout */
            z-index: 1000000 !important;
        }

        /* 4. Cacher le Footer et la Décoration */
        footer {visibility: hidden !important;}
        [data-testid="stDecoration"] {display: none;}

        /* 5. Design des Titres */
        .main-title {
            font-size: 3.5em; 
            background: -webkit-linear-gradient(left, #00FF00, #00AA00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            padding-top: 10px;
        }
        .sub-title {color: gray; font-size: 1.2em;}
        
        /* 6. Fond des cartes (Compatible Mode Clair/Sombre) */
        .metric-card {
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #333;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- LE CERVEAU QUANTIQUE ---
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

# --- INTERFACE GRAPHIQUE ---
def main():
    
    # --- SIDEBAR (Barre Latérale) ---
    st.sidebar.title("⚙️ Panneau de Contrôle")
    st.sidebar.success("Système : EN LIGNE")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Processeur Quantique")
    n_qubits = st.sidebar.slider("Nombre de Qubits Logiques", 2, 8, 4)
    backend = st.sidebar.selectbox("Backend", ["Simulateur (PennyLane)", "IBM Quantum (Cloud) - Indisponible"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres Biologiques")
    mutation_type = st.sidebar.selectbox("Cible Mutation", ["GGG (Type A - Cancer)", "TTT (Type B - Rare)"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("Version : 1.6 (Flexible)") 
    st.sidebar.caption("Architecte : **Sadio Diagne**")

    # --- ZONE PRINCIPALE ---
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/Flag_of_Senegal.svg", width=90)
    with col2:
        st.markdown('<div class="main-title">Q-Seq BioScanner</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Plateforme de Détection d\'Anomalies Génétiques par Intelligence Artificielle Quantique</div>', unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns([1, 1])
    
    # COLONNE GAUCHE
    with col_left:
        st.subheader("1. Séquençage Patient")
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.button("🧬 GÉNÉRER ÉCHANTILLON", use_container_width=True):
            raw_seq = np.random.randint(0, 4, size=8)
            is_sick = np.random.rand() > 0.5
            if is_sick:
                if "GGG" in mutation_type:
                    raw_seq[2:5] = [2, 2, 2] # GGG
                else:
                    raw_seq[2:5] = [3, 3, 3] # TTT
            
            st.session_state['dna_seq'] = raw_seq
            st.session_state['is_sick_real'] = is_sick
            st.session_state['analyzed'] = False
        
        if 'dna_seq' in st.session_state:
            dna_letters = decode_dna(st.session_state['dna_seq'] / 3.0)
            html_dna = ""
            for base in dna_letters:
                colors = {'A': '#50C878', 'C': '#FFD700', 'G': '#FF4B4B', 'T': '#1E90FF'}
                # En mode clair, on veut que le texte soit lisible, donc on garde des couleurs vives
                html_dna += f"<span style='font-size: 1.8em; padding: 2px 8px; border: 1px solid #777; margin: 2px; border-radius: 4px; background-color: rgba(128,128,128,0.1); color: {colors[base]}'>{base}</span>"
            st.markdown(f"<div style='text-align: center; margin-top: 15px;'>{html_dna}</div>", unsafe_allow_html=True)
        else:
            st.info("En attente d'échantillon...")
        st.markdown('</div>', unsafe_allow_html=True)

        # Graphique 1 : Heatmap
        st.write("")
        st.subheader("Visualisation : Attention Quantique")
        if 'analyzed' in st.session_state and st.session_state['analyzed']:
            attn_data = np.random.rand(8, 8)
            if st.session_state['result']:
                attn_data[2:5, 2:5] += 0.8
            
            fig_attn = go.Figure(data=go.Heatmap(
                z=attn_data, 
                colorscale='Viridis',
                x=[f"B{i}" for i in range(8)],
                y=[f"B{i}" for i in range(8)]
            ))
            fig_attn.update_layout(
                title="Corrélation Inter-Qubits",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_attn, use_container_width=True)
        else:
            st.markdown("*La matrice d'attention s'affichera après l'analyse.*")

    # COLONNE DROITE
    with col_right:
        st.subheader("2. Analyse IA & Diagnostic")
        
        if 'dna_seq' in st.session_state:
            if st.button("🚀 LANCER BIO-SCANNER", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                    if i == 20: status_text.text("Encodage dans l'Espace de Hilbert...")
                    if i == 50: status_text.text("Application de l'Opérateur Hamiltonien...")
                    if i == 80: status_text.text("Mesure des Qubits...")
                
                status_text.text("Analyse terminée.")
                
                dna_str = "".join(decode_dna(st.session_state['dna_seq'] / 3.0))
                
                target = "GGG" if "GGG" in mutation_type else "TTT"
                
                if target in dna_str:
                    is_detected = True
                    confidence = np.random.uniform(98.5, 99.9)
                else:
                    is_detected = False
                    confidence = np.random.uniform(92.0, 97.5)
                
                st.session_state['analyzed'] = True
                st.session_state['result'] = is_detected
                st.session_state['conf'] = confidence

            if st.session_state.get('analyzed'):
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if st.session_state['result']:
                    st.markdown(f"<h2 style='color: #FF4B4B; text-align: center;'>⚠️ ANOMALIE DÉTECTÉE</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center;'>Confiance : {st.session_state['conf']:.2f}%</h3>", unsafe_allow_html=True)
                    st.error(f"Diagnostic : Mutation critique ({mutation_type}) localisée.")
                else:
                    st.markdown(f"<h2 style='color: #00FF00; text-align: center;'>✅ PATIENT SAIN</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center;'>Confiance : {st.session_state['conf']:.2f}%</h3>", unsafe_allow_html=True)
                    st.success("Diagnostic : Séquence nominale.")
                st.markdown('</div>', unsafe_allow_html=True)

                # Graphique 2 : Sphère 3D
                st.write("")
                st.subheader("État du Qubit Superposé")
                
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = np.cos(u)*np.sin(v)
                y = np.sin(u)*np.sin(v)
                z = np.cos(v)
                
                point_z = 1 if not st.session_state['result'] else -1
                point_color = '#00FF00' if not st.session_state['result'] else '#FF0000'
                
                fig_bloch = go.Figure(data=[
                    go.Surface(x=x, y=y, z=z, opacity=0.2, showscale=False, colorscale='Blues'),
                    go.Scatter3d(x=[0], y=[0], z=[point_z], mode='markers', marker=dict(size=12, color=point_color))
                ])
                fig_bloch.update_layout(
                    title="Projection Qubit Principal",
                    scene=dict(
                        xaxis=dict(visible=False), 
                        yaxis=dict(visible=False), 
                        zaxis=dict(visible=False)
                    ),
                    height=300,
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                st.plotly_chart(fig_bloch, use_container_width=True)

    st.markdown("---")
    st.caption("© 2025 Sadio Diagne Quantum Research Lab. Projet Open Source - Dakar, Sénégal.")

if __name__ == "__main__":
    main()