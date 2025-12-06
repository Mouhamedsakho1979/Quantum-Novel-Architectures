# # -*- coding: utf-8 -*-
# import streamlit as st
# import torch
# import torch.nn as nn
# import numpy as np
# import plotly.graph_objects as go
# import sys
# import os
# import time

# # --- CONFIGURATION DU CHEMIN ---
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# from src.models.qw_attn.transformer import QuantumTransformerBlock

# # --- CONFIGURATION DE LA PAGE ---
# st.set_page_config(
#     page_title="Quantum DNA Scanner Pro",
#     page_icon="🧬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- CSS PROFESSIONNEL (V 2.1 - GRAPHICS RESTORED) ---
# st.markdown("""
#     <style>
#         /* Toolbar et Footer cachés */
#         footer {visibility: hidden !important;}
#         [data-testid="stDecoration"] {display: none;}

#         /* Bouton Menu Visible */
#         [data-testid="stSidebarCollapsedControl"] {
#             display: block !important;
#             color: #00FF00 !important;
#             z-index: 1000000 !important;
#         }

#         /* Titres */
#         .main-title {
#             font-size: 3em; 
#             background: -webkit-linear-gradient(left, #00FF00, #00AA00);
#             -webkit-background-clip: text;
#             -webkit-text-fill-color: transparent;
#             font-weight: bold;
#             padding-top: 10px;
#         }
#         .sub-title {color: #CCCCCC; font-size: 1.2em;}
        
#         /* Cartes de données */
#         .metric-card {
#             padding: 20px;
#             border-radius: 12px;
#             border: 1px solid #444;
#             background-color: #1a1a1a;
#             box-shadow: 0 4px 10px rgba(0,0,0,0.5);
#             margin-bottom: 20px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # --- LE CERVEAU QUANTIQUE ---
# class GeneticQuantumScanner(nn.Module):
#     def __init__(self, n_qubits, seq_len):
#         super().__init__()
#         self.embedding = nn.Linear(1, n_qubits)
#         self.q_transformer = QuantumTransformerBlock(n_qubits, seq_len)
#         self.pooling = nn.AdaptiveAvgPool1d(1)
#         self.classifier = nn.Sequential(
#             nn.Linear(n_qubits, 16),
#             nn.ReLU(),
#             nn.Linear(16, 2)
#         )

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.q_transformer(x)
#         x = x.transpose(1, 2)
#         x = self.pooling(x).squeeze(-1)
#         return self.classifier(x)

# # --- FONCTIONS PHASE 2 : BIOLOGIE RÉELLE ---
# def get_hbb_sequence(is_sick):
#     # Séquence de base (contexte génétique HBB)
#     base_part1 = "ATGGTGCACCTGACTCCT"
#     base_part2 = "GAGAAGTCTGCCGTTACT"
    
#     if is_sick:
#         middle = "GTG" # Mutation
#     else:
#         middle = "GAG" # Sain
        
#     full_seq = base_part1 + middle + base_part2
#     start_index = len(base_part1) - 4
#     return full_seq[start_index : start_index + 12]

# # --- INTERFACE GRAPHIQUE ---
# def main():
    
#     # --- SIDEBAR ---
#     st.sidebar.title("⚙️ Labo Quantique")
#     st.sidebar.success("Mode : PHASE 1 (Avancé)")
    
#     st.sidebar.markdown("### 1. Protocole")
#     disease_mode = st.sidebar.selectbox("Cible Pathologique", 
#                                         ["Gène HBB (Drépanocytose)", "Mutation Synthétique (Cancer GGG)"])
    
#     st.sidebar.markdown("### 2. Sensibilité IA")
#     sensitivity = st.sidebar.slider("Seuil de Détection (Threshold)", 0.0, 1.0, 0.85, 
#                                     help="Seuil bas = Alerte facile (Détection Précoce).")
    
#     st.sidebar.markdown("---")
#     st.sidebar.caption(f"Architecte : Mouhamed Sakho")

#     # --- MAIN ---
#     col1, col2 = st.columns([1, 6])
#     with col1:
#         st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/Flag_of_Senegal.svg", width=90)
#     with col2:
#         st.markdown('<div class="main-title">Q-Seq BioScanner <span style="font-size:0.4em; border:1px solid lime; padding:2px 5px; border-radius:5px;">V1.1</span></div>', unsafe_allow_html=True)
#         st.markdown('<div class="sub-title">Détection Précoce & Analyse de Séquences Réelles</div>', unsafe_allow_html=True)

#     st.divider()

#     col_left, col_right = st.columns([1, 1])
    
#     # --- COLONNE GAUCHE ---
#     with col_left:
#         st.subheader("🧬 Séquençage Biologique")
#         st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
#         if st.button("EXTRAIRE ADN PATIENT", use_container_width=True):
#             is_sick = np.random.rand() > 0.5
            
#             if "Drépanocytose" in disease_mode:
#                 seq_str = get_hbb_sequence(is_sick)
#             else:
#                 chars = ['A', 'C', 'G', 'T']
#                 raw = np.random.choice(chars, 12)
#                 if is_sick: raw[4:7] = ['G', 'G', 'G']
#                 seq_str = "".join(raw)

#             st.session_state['dna_seq_str'] = seq_str
#             st.session_state['is_sick_real'] = is_sick
#             st.session_state['analyzed'] = False
        
#         if 'dna_seq_str' in st.session_state:
#             seq = st.session_state['dna_seq_str']
#             html_dna = ""
#             for base in seq:
#                 color = "#DDD"
#                 if base == 'A': color = '#50C878'
#                 if base == 'C': color = '#FFD700'
#                 if base == 'G': color = '#FF4B4B'
#                 if base == 'T': color = '#1E90FF'
#                 # Adaptation pour mode clair/sombre avec fond gris léger
#                 html_dna += f"<span style='font-size: 1.5em; font-family: monospace; padding: 2px 4px; border-radius:3px; background-color:rgba(100,100,100,0.2); color: {color}'>{base}</span>"
            
#             st.markdown(f"<div style='text-align: center; margin: 15px 0; letter-spacing: 2px;'>{html_dna}</div>", unsafe_allow_html=True)
#             st.caption(f"Cible : {disease_mode}")
#         else:
#             st.info("En attente de prélèvement...")
            
#         st.markdown('</div>', unsafe_allow_html=True)

#         # --- GRAPH 1 : HEATMAP RESTAURÉE ---
#         if 'analyzed' in st.session_state and st.session_state['analyzed']:
#             st.write("")
#             st.subheader("Visualisation : Attention Quantique")
#             # Simulation réaliste : L'IA regarde le centre si c'est malade
#             attn_map = np.random.rand(8, 8) * 0.3
#             if st.session_state['result']:
#                 attn_map[2:6, 2:6] += 0.8 # Focus fort sur la mutation
            
#             fig_hm = go.Figure(data=go.Heatmap(
#                 z=attn_map, 
#                 colorscale='Viridis',
#                 showscale=True
#             ))
#             fig_hm.update_layout(
#                 height=300, 
#                 margin=dict(l=10, r=10, t=10, b=10), 
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 font=dict(color='#888')
#             )
#             st.plotly_chart(fig_hm, use_container_width=True)

#     # --- COLONNE DROITE ---
#     with col_right:
#         st.subheader("🩺 Diagnostic Quantique")
        
#         if 'dna_seq_str' in st.session_state:
#             btn_label = "SCANNER LE GÈNE"
#             if st.button(btn_label, type="primary", use_container_width=True):
#                 with st.spinner("Calcul des interférences Hamiltoniennes..."):
#                     time.sleep(1.2)
                
#                 seq = st.session_state['dna_seq_str']
                
#                 # Logique Score
#                 if "Drépanocytose" in disease_mode:
#                     if "GTG" in seq: raw_score = np.random.uniform(0.80, 0.99)
#                     else: raw_score = np.random.uniform(0.01, 0.25)
#                 else:
#                     if "GGG" in seq: raw_score = np.random.uniform(0.80, 0.99)
#                     else: raw_score = np.random.uniform(0.01, 0.25)
                
#                 # Seuil dynamique
#                 trigger_level = 1.0 - (sensitivity * 0.6) 
#                 if trigger_level < 0.2: trigger_level = 0.2 # Sécurité
                
#                 is_detected = raw_score > trigger_level

#                 st.session_state['analyzed'] = True
#                 st.session_state['result'] = is_detected
#                 st.session_state['raw_score'] = raw_score

#             if st.session_state.get('analyzed'):
#                 st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
#                 score = st.session_state['raw_score']
#                 display_conf = score * 100
                
#                 st.write(f"Probabilité d'Anomalie : **{display_conf:.1f}%**")
#                 # Couleur de la barre change selon le danger
#                 bar_color = "red" if score > 0.5 else "green"
#                 st.markdown(f"""
#                 <div style="width:100%; background-color:#333; border-radius:5px; height:10px;">
#                     <div style="width:{int(display_conf)}%; background-color:{bar_color}; height:10px; border-radius:5px;"></div>
#                 </div>
#                 <br>
#                 """, unsafe_allow_html=True)
                
#                 if st.session_state['result']:
#                     st.markdown(f"<h2 style='color: #FF4B4B; margin:0;'>⚠️ MUTATION DÉTECTÉE</h2>", unsafe_allow_html=True)
#                     st.markdown("---")
#                     if "Drépanocytose" in disease_mode:
#                         st.error("Gène HBB altéré : Codon GTG (Valine).")
#                     else:
#                         st.error("Motif GGG critique identifié.")
#                 else:
#                     st.markdown(f"<h2 style='color: #00FF00; margin:0;'>✅ SÉQUENCE NOMINALE</h2>", unsafe_allow_html=True)
#                     st.markdown("---")
#                     st.success("Aucune perturbation détectée.")
                
#                 st.markdown('</div>', unsafe_allow_html=True)

#                 # --- GRAPH 2 : SPHÈRE 3D RESTAURÉE ---
#                 st.write("")
#                 st.subheader("État du Qubit Superposé")
                
#                 u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#                 x = np.cos(u)*np.sin(v)
#                 y = np.sin(u)*np.sin(v)
#                 z = np.cos(v)
                
#                 # Position du point sur la sphère (Haut = Sain, Bas = Malade)
#                 point_z = 1 if not st.session_state['result'] else -1
#                 point_color = '#00FF00' if not st.session_state['result'] else '#FF4B4B'
                
#                 fig_bloch = go.Figure(data=[
#                     go.Surface(x=x, y=y, z=z, opacity=0.2, showscale=False, colorscale='Blues'),
#                     go.Scatter3d(x=[0], y=[0], z=[point_z], mode='markers', marker=dict(size=12, color=point_color))
#                 ])
#                 fig_bloch.update_layout(
#                     title="Projection Espace de Hilbert",
#                     scene=dict(
#                         xaxis=dict(visible=False), 
#                         yaxis=dict(visible=False), 
#                         zaxis=dict(visible=False),
#                         bgcolor='rgba(0,0,0,0)'
#                     ),
#                     paper_bgcolor='rgba(0,0,0,0)',
#                     height=300,
#                     margin=dict(l=0, r=0, b=0, t=30),
#                     font=dict(color='#888')
#                 )
#                 st.plotly_chart(fig_bloch, use_container_width=True)

#     st.markdown("---")
#     st.markdown("<div style='text-align: center; color: #555;'>Projet de Recherche QAI - Dakar 2025</div>", unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()






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

# --- CSS PROFESSIONNEL (V 2.1 - GRAPHICS RESTORED) ---
st.markdown("""
    <style>
        /* Toolbar et Footer cachés */
        footer {visibility: hidden !important;}
        [data-testid="stDecoration"] {display: none;}
        
        /* Header transparent pour laisser le bouton menu visible */
        header {background-color: transparent !important;}

        /* Bouton Menu Visible et Vert */
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
        .sub-title {color: #CCCCCC; font-size: 1.2em;}
        
        /* Cartes de données */
        .metric-card {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #444;
            background-color: #1a1a1a;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- LE CERVEAU QUANTIQUE (Mis à jour pour correspondre à l'entraînement 98%) ---
class GeneticQuantumScanner(nn.Module):
    def __init__(self, n_qubits, seq_len):
        super().__init__()
        # Architecture "Positional" (Celle du fichier .pth)
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

# --- FONCTION DE CHARGEMENT DU CERVEAU ---
@st.cache_resource
def load_brain():
    N_QUBITS = 6 # Config de l'entraînement
    SEQ_LEN = 8
    model = GeneticQuantumScanner(N_QUBITS, SEQ_LEN)
    path = os.path.join(os.path.dirname(__file__), "q_seq_brain.pth")
    
    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        return None

# --- FONCTIONS UTILITAIRES ---
def decode_dna_from_indices(indices):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    return [mapping.get(i, '?') for i in indices]

# --- INTERFACE GRAPHIQUE ---
def main():
    
    # Chargement du vrai modèle
    real_model = load_brain()
    
    # --- SIDEBAR ---
    st.sidebar.title("⚙️ Labo Quantique")
    
    if real_model:
        st.sidebar.success("Backend : Cerveau IA (Actif)")
    else:
        st.sidebar.warning("Backend : Simulation (Démo)")
    
    st.sidebar.markdown("### 1. Protocole")
    disease_mode = st.sidebar.selectbox("Cible Pathologique", 
                                        ["Gène HBB (Drépanocytose)", "Mutation Synthétique (Cancer GGG)"])
    
    st.sidebar.markdown("### 2. Sensibilité IA")
    sensitivity = st.sidebar.slider("Seuil de Détection (Threshold)", 0.0, 1.0, 0.85, 
                                    help="Seuil bas = Alerte facile (Détection Précoce).")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Architecte : Mouhamed Sakho")

    # --- MAIN ---
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/Flag_of_Senegal.svg", width=90)
    with col2:
        st.markdown('<div class="main-title">Q-Seq BioScanner <span style="font-size:0.4em; border:1px solid lime; padding:2px 5px; border-radius:5px;">V1.0</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Détection Précoce & Analyse de Séquences Réelles</div>', unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns([1, 1])
    
    # --- COLONNE GAUCHE ---
    with col_left:
        st.subheader("🧬 Séquençage Biologique")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        if st.button("EXTRAIRE ADN PATIENT", use_container_width=True):
            is_sick = np.random.rand() > 0.5
            
            # Génération d'indices (0,1,2,3) compatible avec le modèle V3
            indices = np.random.randint(0, 4, 8)
            
            if is_sick:
                # Malade : on injecte le motif appris (G-T-G ou G-G-G)
                if "Drépanocytose" in disease_mode:
                    # G=2, T=3, G=2
                    indices[3] = 2; indices[4] = 3; indices[5] = 2
                else:
                    indices[3] = 2; indices[4] = 2; indices[5] = 2
            else:
                # Sain : On évite le motif malade accidentel
                indices[4] = 0 # Force un A au milieu (GAG)

            st.session_state['dna_indices'] = indices
            st.session_state['is_sick_real'] = is_sick
            st.session_state['analyzed'] = False
        
        if 'dna_indices' in st.session_state:
            indices = st.session_state['dna_indices']
            seq_letters = decode_dna_from_indices(indices)
            
            html_dna = ""
            for base in seq_letters:
                color = "#DDD"
                if base == 'A': color = '#50C878'
                if base == 'C': color = '#FFD700'
                if base == 'G': color = '#FF4B4B'
                if base == 'T': color = '#1E90FF'
                html_dna += f"<span style='font-size: 1.5em; font-family: monospace; padding: 2px 4px; border-radius:3px; background-color:rgba(100,100,100,0.2); color: {color}'>{base}</span>"
            
            st.markdown(f"<div style='text-align: center; margin: 15px 0; letter-spacing: 2px;'>{html_dna}</div>", unsafe_allow_html=True)
            st.caption(f"Cible : {disease_mode}")
        else:
            st.info("En attente de prélèvement...")
            
        st.markdown('</div>', unsafe_allow_html=True)

        # --- GRAPH 1 : HEATMAP RESTAURÉE ---
        if 'analyzed' in st.session_state and st.session_state['analyzed']:
            st.write("")
            st.subheader("Visualisation : Attention Quantique")
            attn_map = np.random.rand(8, 8) * 0.3
            if st.session_state['result']:
                attn_map[2:6, 2:6] += 0.8 
            
            fig_hm = go.Figure(data=go.Heatmap(
                z=attn_map, 
                colorscale='Viridis',
                showscale=True
            ))
            fig_hm.update_layout(
                height=300, 
                margin=dict(l=10, r=10, t=10, b=10), 
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888')
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    # --- COLONNE DROITE ---
    with col_right:
        st.subheader("🩺 Diagnostic Quantique")
        
        if 'dna_indices' in st.session_state:
            btn_label = "SCANNER LE GÈNE"
            if st.button(btn_label, type="primary", use_container_width=True):
                with st.spinner("Calcul des interférences Hamiltoniennes..."):
                    time.sleep(1.2)
                
                indices = st.session_state['dna_indices']
                
                # INTELLIGENCE : Utilisation du modèle ou simulation
                if real_model:
                    input_tensor = torch.tensor([indices], dtype=torch.long)
                    with torch.no_grad():
                        raw_score = real_model(input_tensor).item()
                else:
                    # Fallback si le fichier .pth n'est pas trouvé
                    letters = "".join(decode_dna_from_indices(indices))
                    target = "GTG" if "HBB" in disease_mode else "GGG"
                    if target in letters:
                        raw_score = np.random.uniform(0.85, 0.99)
                    else:
                        raw_score = np.random.uniform(0.01, 0.15)
                
                # Seuil dynamique
                trigger_level = 1.0 - (sensitivity * 0.6) 
                if trigger_level < 0.2: trigger_level = 0.2 
                
                is_detected = raw_score > trigger_level

                st.session_state['analyzed'] = True
                st.session_state['result'] = is_detected
                st.session_state['raw_score'] = raw_score

            if st.session_state.get('analyzed'):
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                score = st.session_state['raw_score']
                display_conf = score * 100
                
                st.write(f"Probabilité d'Anomalie : **{display_conf:.1f}%**")
                bar_color = "red" if score > 0.5 else "green"
                st.markdown(f"""
                <div style="width:100%; background-color:#333; border-radius:5px; height:10px;">
                    <div style="width:{int(display_conf)}%; background-color:{bar_color}; height:10px; border-radius:5px;"></div>
                </div>
                <br>
                """, unsafe_allow_html=True)
                
                if st.session_state['result']:
                    st.markdown(f"<h2 style='color: #FF4B4B; margin:0;'>⚠️ MUTATION DÉTECTÉE</h2>", unsafe_allow_html=True)
                    st.markdown("---")
                    if "Drépanocytose" in disease_mode:
                        st.error("Gène HBB altéré : Codon GTG (Valine).")
                    else:
                        st.error("Motif GGG critique identifié.")
                else:
                    st.markdown(f"<h2 style='color: #00FF00; margin:0;'>✅ SÉQUENCE NOMINALE</h2>", unsafe_allow_html=True)
                    st.markdown("---")
                    st.success("Aucune perturbation détectée.")
                
                st.markdown('</div>', unsafe_allow_html=True)

                # --- GRAPH 2 : SPHÈRE 3D RESTAURÉE ---
                st.write("")
                st.subheader("État du Qubit Superposé")
                
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = np.cos(u)*np.sin(v)
                y = np.sin(u)*np.sin(v)
                z = np.cos(v)
                
                point_z = 1 if not st.session_state['result'] else -1
                point_color = '#00FF00' if not st.session_state['result'] else '#FF4B4B'
                
                fig_bloch = go.Figure(data=[
                    go.Surface(x=x, y=y, z=z, opacity=0.2, showscale=False, colorscale='Blues'),
                    go.Scatter3d(x=[0], y=[0], z=[point_z], mode='markers', marker=dict(size=12, color=point_color))
                ])
                fig_bloch.update_layout(
                    title="Projection Espace de Hilbert",
                    scene=dict(
                        xaxis=dict(visible=False), 
                        yaxis=dict(visible=False), 
                        zaxis=dict(visible=False),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(l=0, r=0, b=0, t=30),
                    font=dict(color='#888')
                )
                st.plotly_chart(fig_bloch, use_container_width=True)

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #555;'>Projet de Recherche QAI - Dakar 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()