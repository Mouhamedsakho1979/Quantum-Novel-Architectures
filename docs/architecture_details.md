# Manuel Technique et Guide d'Exécution

Ce document détaille le fonctionnement et les résultats attendus pour chaque architecture du projet.

---

# 🟢 1. QFCC – Quantum Feature Cascade Classifier

### C’est quoi ?

Un classifieur quantique qui apprend progressivement via une cascade de circuits.

### Pourquoi c'est nouveau ?

Architecture modulaire, encodage angulaire optimisé.

### Lancer :

```bash
python3 src/experiments/run_milestone1_qfcc.py
```

### Résultats attendus :

* Accuracy entre **80% et 90%**
* Courbe de loss descendante

---

# 🔵 2. Q-AGD – Quantum Adaptive Gradient Descent

### Concept :

Optimisation sans backpropagation grâce à la superposition.

### Lancer :

```bash
python3 src/experiments/run_milestone2_qagd.py
```

### Résultats attendus :

* Convergence rapide vers `[1.0, 1.0]`
* Courbe en chute brutale

---

# 🟣 3. SL-QGAN – Stabilized Layered QGAN

### Concept :

Un GAN quantique stabilisé par couches.

### Lancer :

```bash
python3 src/experiments/run_milestone3_qgan.py
```

### Résultats attendus :

* Points générés formant un cercle

---

# 🟠 4. QW-Attn – Quantum Transformer

### Concept :

Attention quantique via un overlapped interference test.

### Lancer :

```bash
python3 src/experiments/run_milestone4_transformer.py
```

### Résultats attendus :

* MSE proche de 0

---

# 🔴 5. HRQN – Hamiltonian Residual Quantum Network

### Concept :

Évolution naturelle sous Hamiltonien pour approximer des fonctions.

### Lancer :

```bash
python3 src/experiments/run_milestone5_hrqn.py
```

### Résultats attendus :

* Courbe violette épousant parfaitement la courbe cible

---

# 🧬 Projet Final : Q-Seq – Quantum Sequence Classifier

### Concept :

Prototype de produit basé sur un Transformer quantique appliqué à l’ADN synthétique.

### Lancer :

```bash
python3 src/experiments/run_project_QSeq.py
```

