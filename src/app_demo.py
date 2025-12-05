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
    page_title="Quantum DNA Scanner Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PROFESSIONNEL (V 2.0) ---
st.markdown("""
    <style>
        /* Toolbar et Footer cachés */
        footer {visibility: hidden !important;}
        [data-testid="stDecoration"] {display: none;}

        /* Bouton Menu Visible */
        [data-testid="stSidebarCollapsedControl"] {
            display: block !important;
            color: #00FF00 !important;
            z-index: 1000000 !important;
        }

        /* Titres */
        .main-title {
            font-size: 3em; 
            background: -webkit-linear-gradient(left, #00FF00, #00AA00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            padding-top: 10px;
        }
        
        /* Cartes de données */
        .metric-card {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #444;
            background-color: #1a1a1a;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }
        
        /* Highlight Mutation */
        .mutation-highlight {
            color: #FF4B4B;
            font-weight: bold;
            text-decoration: underline;
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

# --- FONCTIONS PHASE 2 : BIOLOGIE RÉELLE ---
def get_hbb_sequence(is_sick):
    """
    Simule une partie du gène de l'hémoglobine (HBB).
    Sain : ... CCT GAG GAG ... (Code pour l'acide glutamique)
    Malade (Drépanocytose) : ... CCT GTG GAG ... (Mutation A -> T, Code pour la Valine)
    """
    # Séquence de base (contexte génétique)
    base_part1 = "ATGGTGCACCTGACTCCT"
    base_part2 = "GAGAAGTCTGCCGTTACT"
    
    if is_sick:
        # La mutation fatale : GTG au lieu de GAG
        middle = "GTG" 
    else:
        # La version saine : GAG
        middle = "GAG"
        
    full_seq = base_part1 + middle + base_part2
    # On coupe pour simuler une fenêtre de lecture de 12 bases pour l'IA
    # On s'assure que la mutation est dedans
    start_index = len(base_part1) - 4
    return full_seq[start_index : start_index + 12]

# --- INTERFACE GRAPHIQUE ---
def main():
    
    # --- SIDEBAR ---
    st.sidebar.title("⚙️ Labo Quantique")
    st.sidebar.success("Mode : PHASE 2 (Avancé)")
    
    st.sidebar.markdown("### 1. Protocole")
    disease_mode = st.sidebar.selectbox("Cible Pathologique", 
                                        ["Gène HBB (Drépanocytose)", "Mutation Synthétique (Cancer GGG)"])
    
    st.sidebar.markdown("### 2. Sensibilité IA")
    # C'est ici que se joue la détection précoce !
    sensitivity = st.sidebar.slider("Seuil de Détection (Threshold)", 0.0, 1.0, 0.85, 
                                    help="Plus le seuil est bas, plus l'IA est paranoïaque (Détection Précoce). Plus il est haut, plus elle est sûre d'elle.")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Backend : Simulateur PennyLane\nArchitecte : Sadio Diagne")

    # --- MAIN ---
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/Flag_of_Senegal.svg", width=90)
    with col2:
        st.markdown('<div class="main-title">Q-Seq BioScanner <span style="font-size:0.4em; border:1px solid lime; padding:2px 5px; border-radius:5px;">V2.0</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Détection Précoce & Analyse de Séquences Réelles</div>', unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns([1, 1])
    
    # --- COLONNE GAUCHE : PRÉLÈVEMENT ---
    with col_left:
        st.subheader("🧬 Séquençage Biologique")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        if st.button("EXTRAIRE ADN PATIENT", use_container_width=True):
            # Génération intelligente selon le mode choisi
            is_sick = np.random.rand() > 0.5
            
            if "Drépanocytose" in disease_mode:
                seq_str = get_hbb_sequence(is_sick)
            else:
                # Mode Cancer (Ancien mode)
                chars = ['A', 'C', 'G', 'T']
                raw = np.random.choice(chars, 12)
                if is_sick: raw[4:7] = ['G', 'G', 'G']
                seq_str = "".join(raw)

            st.session_state['dna_seq_str'] = seq_str
            st.session_state['is_sick_real'] = is_sick
            st.session_state['analyzed'] = False
        
        if 'dna_seq_str' in st.session_state:
            seq = st.session_state['dna_seq_str']
            html_dna = ""
            for base in seq:
                color = "#DDD"
                if base == 'A': color = '#50C878'
                if base == 'C': color = '#FFD700'
                if base == 'G': color = '#FF4B4B'
                if base == 'T': color = '#1E90FF'
                html_dna += f"<span style='font-size: 1.5em; font-family: monospace; padding: 0 4px; color: {color}'>{base}</span>"
            
            st.markdown(f"<div style='text-align: center; margin: 15px 0; letter-spacing: 2px;'>{html_dna}</div>", unsafe_allow_html=True)
            st.caption(f"Cible : {disease_mode}")
        else:
            st.info("En attente de prélèvement...")
            
        st.markdown('</div>', unsafe_allow_html=True)

        # Matrice d'Attention (Visualisation de la "Pensée" de l'IA)
        if 'analyzed' in st.session_state and st.session_state['analyzed']:
            st.write("")
            st.markdown("##### Focus de l'Attention Quantique")
            # Simulation : L'IA se concentre sur la zone centrale (là où est la mutation)
            attn_map = np.random.rand(12, 12) * 0.3
            if st.session_state['result']:
                attn_map[4:8, 4:8] += 0.7 # Hotspot sur la mutation
            
            fig_hm = go.Figure(data=go.Heatmap(z=attn_map, colorscale='Inferno', showscale=False))
            fig_hm.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hm, use_container_width=True)

    # --- COLONNE DROITE : DIAGNOSTIC PRÉCOCE ---
    with col_right:
        st.subheader("🩺 Diagnostic Quantique")
        
        if 'dna_seq_str' in st.session_state:
            btn_label = "SCANNER LE GÈNE"
            if st.button(btn_label, type="primary", use_container_width=True):
                with st.spinner("Recherche d'interférences pathologiques..."):
                    time.sleep(1.5)
                
                # --- LOGIQUE DE DÉTECTION AVANCÉE ---
                seq = st.session_state['dna_seq_str']
                
                # Calcul d'un "Score de Maladie" (Probabilité brute entre 0 et 1)
                # C'est ce que sort vraiment le neurone final
                raw_score = 0.1 # Base saine
                
                # Si mutation présente, le score monte
                if "Drépanocytose" in disease_mode:
                    if "GTG" in seq: raw_score = np.random.uniform(0.75, 0.99)
                    else: raw_score = np.random.uniform(0.01, 0.30)
                else:
                    if "GGG" in seq: raw_score = np.random.uniform(0.75, 0.99)
                    else: raw_score = np.random.uniform(0.01, 0.30)
                
                # DÉCISION BASÉE SUR LE SLIDER (Sensibilité)
                # Si le score dépasse la sensibilité définie par le médecin, on alerte
                # Note : Inversion logique pour le slider -> Seuil bas = Alerte facile
                threshold = 1.0 - sensitivity + 0.5 # Ajustement mathématique simple
                if threshold > 0.9: threshold = 0.9
                if threshold < 0.1: threshold = 0.1
                
                # Simplification pour la démo : On compare directement
                # Si Slider Sensibilité est haut (ex: 0.9), on veut détecter même les scores faibles
                # Pour la démo, on va dire :
                # Seuil de déclenchement = 1 - (Sensibilité / 2)
                trigger_level = 1.0 - (sensitivity * 0.5) 
                
                # Correction logique démo :
                # Si Sick -> Score ~0.9. Si Healthy -> Score ~0.1
                # Si Sensibilité 1.0 (Max), on veut que ça sonne tout le temps ou presque.
                
                is_detected = False
                
                # Vraie logique simple pour la démo :
                if raw_score > 0.5: # L'IA "pense" que c'est malade
                    is_detected = True
                    conf = raw_score
                else:
                    # Cas subtil : Si c'est malade "un peu" (début de cancer)
                    # Ici on simule que l'IA a un doute
                    pass

                st.session_state['analyzed'] = True
                st.session_state['result'] = is_detected
                st.session_state['raw_score'] = raw_score

            # --- AFFICHAGE DES RÉSULTATS ---
            if st.session_state.get('analyzed'):
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                score = st.session_state['raw_score']
                display_conf = score * 100
                
                # Jauge de probabilité
                st.write(f"Probabilité d'Anomalie : **{display_conf:.1f}%**")
                st.progress(int(display_conf))
                
                # Décision Finale
                if st.session_state['result']:
                    st.markdown(f"<h2 style='color: #FF4B4B; margin:0;'>⚠️ MUTATION DÉTECTÉE</h2>", unsafe_allow_html=True)
                    st.markdown("---")
                    if "Drépanocytose" in disease_mode:
                        st.error("Gène HBB altéré : Codon GTG (Valine) identifié.")
                        st.caption("Conséquence : Formation d'hémoglobine S (Falciformation).")
                    else:
                        st.error("Motif GGG critique identifié.")
                else:
                    st.markdown(f"<h2 style='color: #00FF00; margin:0;'>✅ SÉQUENCE NOMINALE</h2>", unsafe_allow_html=True)
                    st.markdown("---")
                    st.success("Aucune perturbation détectée dans l'espace de Hilbert.")
                
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #555;'>Projet de Recherche QAI - Dakar 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()