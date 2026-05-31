#  Localisation par Trilatération

Projet du cours **UM4MA266 — Optimisation numérique et science des données** (Master Mathématiques, Sorbonne Université).

Implémentation et comparaison de plusieurs algorithmes d'optimisation pour résoudre le problème de localisation par trilatération, inspiré du principe de fonctionnement du GPS.

---

## Problème

Étant données les positions connues de $m$ stations $(\mathbf{a}_i)$ et les distances mesurées $(d_i)$, on cherche la position inconnue $\mathbf{x} \in \mathbb{R}^n$ minimisant :

$$\min_{\mathbf{x} \in \mathbb{R}^n} \sum_{i=1}^{m} \left( \|\mathbf{x} - \mathbf{a}_i\| - d_i \right)^2$$

En pratique, les distances sont **bruitées** et on dispose souvent de plus de 3 mesures (multilatération).

---

## Structure du notebook

### Question 1 — Moindres carrés non-linéaires

Implémentation from scratch de deux algorithmes d'optimisation via **PyTorch** (autodiff pour le calcul du jacobien) :

- **Gauss-Newton** : résolution itérative du système normal $J^T J \delta = J^T r$, avec fallback sur la pseudo-inverse en cas de matrice mal conditionnée.
- **Levenberg-Marquardt** : variante robuste avec paramètre de régularisation $\lambda$ adaptatif.

Comparaison avec deux méthodes de **descente de gradient** :
- Gradient à **pas fixe**
- Gradient à **pas optimal** (recherche de ligne par `fsolve`)

**Étude de sensibilité au point initial** : tests sur plusieurs points de départ pour analyser la convergence vers des minima locaux.

### Question 2 — Comparaison géométrique des algorithmes

Visualisation des trajectoires de convergence superposées aux courbes de niveau de la fonction objectif et aux cercles de distance des stations. Mise en évidence des comportements différents selon la géométrie du problème.

### Question 3 — Relaxation SDP avec gradient projeté

Formulation du problème avec contrainte semi-définie positive : en introduisant $X = \mathbf{x}\mathbf{x}^T$, les contraintes quadratiques deviennent linéaires en $(\mathbf{x}, X)$.

Deux approches testées :
- **Gradient projeté à pas fixe** : descente + projection sur le cône SDP à chaque itération.
- **Gradient projeté à pas optimal** : recherche du pas optimal à chaque étape.

### Question 4 — Influence du bruit et de la géométrie des stations

Tests systématiques sur différentes configurations de stations avec des niveaux de bruit croissants (`noise_std` de 0.01 à 2) :

| Configuration | Observation |
|---|---|
| Stations bien réparties (carré) | Tous les algorithmes convergent, erreur croît régulièrement avec le bruit |
| Stations alignées | Problème mal conditionné — gradient non projeté très dégradé, gradient projeté nettement meilleur |
| Stations regroupées (même secteur) | Perte de précision pour tous les algorithmes |

### Question 5 — Extension GPS : biais d'horloge inconnu

Dans le cas GPS, les mesures sont des **pseudo-distances** avec un biais d'horloge $b$ inconnu :

$$c \cdot \tau_i = \|\mathbf{x} - \mathbf{a}_i\| + b$$

On étend le vecteur d'inconnues à $(\mathbf{x}, b) \in \mathbb{R}^{n+1}$ et on adapte tous les algorithmes (Gauss-Newton, Levenberg-Marquardt, descentes de gradient, gradient projeté SDP).

**Résultats :**
- Levenberg-Marquardt : le plus robuste au choix du point initial.
- Gradient à pas fixe : peu sensible au point de départ, légèrement moins précis que LM.
- Gradient à pas optimal : convergence irrégulière dans ce cas.
- Gradient projeté SDP (fixe et optimal) : convergence lente, n'atteint pas précisément la solution.

---

## Installation

```bash
pip install torch numpy scipy matplotlib
```

---

## Utilisation

```bash
jupyter notebook projet_1__1___1_.ipynb
```

Aucun dataset externe requis — toutes les données sont **simulées** via les fonctions `generate_trilateration_data` et `generate_gps_pseudoranges`.

---

## Dépendances

| Bibliothèque | Usage |
|---|---|
| `torch` | Calcul du jacobien par autodiff, opérations vectorielles |
| `numpy` | Algèbre linéaire, génération des données simulées |
| `scipy.optimize` | `fsolve` pour la recherche de pas optimal |
| `matplotlib` | Visualisation des trajectoires et surfaces |

---

## Références

- Boyd & Vandenberghe, *Convex Optimization* — https://web.stanford.edu/~boyd/cvxbook/
- Wikipedia : [Trilateration](https://en.wikipedia.org/wiki/Trilateration), [GPS](https://en.wikipedia.org/wiki/Global_Positioning_System)
