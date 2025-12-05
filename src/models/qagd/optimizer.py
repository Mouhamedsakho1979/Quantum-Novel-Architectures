import torch
from torch.optim import Optimizer

class QAGD(Optimizer):
    """
    Quantum Adaptive Gradient Descent (Q-AGD)
    Implémentation inspirée de SPSA (Simultaneous Perturbation Stochastic Approximation).
    
    Idée clé : Au lieu de calculer le gradient exact (coûteux sur machine quantique),
    on perturbe tous les paramètres simultanément dans une superposition simulée
    pour estimer la meilleure direction de descente avec seulement 2 évaluations.
    """
    def __init__(self, params, lr=0.1, perturbation=0.1, alpha=0.602, gamma=0.101):
        defaults = dict(lr=lr, perturbation=perturbation, alpha=alpha, gamma=gamma)
        super(QAGD, self).__init__(params, defaults)
        self.k = 0  # Compteur d'itérations

    def step(self, closure=None):
        """
        Effectue une étape d'optimisation.
        closure : Une fonction qui recalcule le modèle et retourne la loss.
        """
        if closure is None:
            raise RuntimeError("Q-AGD nécessite une closure pour évaluer la loss (fonction de coût).")

        loss = None
        self.k += 1

        for group in self.param_groups:
            # Hyperparamètres adaptatifs (Decreasing gain sequence)
            c_k = group['perturbation'] / (self.k ** group['gamma'])
            a_k = group['lr'] / (self.k ** group['alpha'])

            # Sauvegarde des paramètres actuels
            params = []
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)

            # 1. Génération du vecteur de perturbation (Delta)
            # Simule la superposition : +1 ou -1 aléatoire (Bernoulli)
            deltas = [torch.randint(0, 2, p.shape, device=p.device).float() * 2 - 1 for p in params]

            # 2. Perturbation Positive (+ c_k * delta)
            for p, delta in zip(params, deltas):
                p.data.add_(delta, alpha=c_k)
            
            loss_plus = closure() # Évaluation f(theta + delta)

            # 3. Perturbation Négative (- 2 * c_k * delta pour arriver à theta - c_k * delta)
            for p, delta in zip(params, deltas):
                p.data.sub_(delta, alpha=2 * c_k)
            
            loss_minus = closure() # Évaluation f(theta - delta)

            # 4. Restauration des paramètres et Mise à jour (Update)
            # Gradient estimé g_k = (f+ - f-) / (2ck) * delta
            grad_estimate_scalar = (loss_plus - loss_minus) / (2 * c_k)

            for p, delta in zip(params, deltas):
                # On revient au point central
                p.data.add_(delta, alpha=c_k) 
                # Mise à jour des poids : theta = theta - a_k * grad_estimate
                p.data.sub_(delta, alpha=a_k * grad_estimate_scalar)
            
            # On stocke la loss actuelle pour le suivi
            loss = loss_plus 

        return loss