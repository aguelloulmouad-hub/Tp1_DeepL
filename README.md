# TP Deep Learning – Optimisation des réseaux Feedforward profonds

## Description

Ce dépôt contient les scripts Python réalisés dans le cadre du Travaux Pratiques **« Optimisation des Réseaux Feedforward Profonds »** du Master IA.  
L’objectif est de prédire le prix médian des maisons en Californie (jeu de données `California Housing` de scikit-learn) à l’aide d’un MLP (perceptron multicouche) intégrant des techniques avancées de régularisation et d’optimisation.

Les principaux thèmes abordés sont :

- Exploration et prétraitement des données (standardisation, split stratifié)
- Implémentation modulaire d’un MLP avec BatchNorm, Dropout, initialisation adaptée
- Boucle d’entraînement avec gradient clipping, scheduler `ReduceLROnPlateau` et early stopping
- Étude d’ablation (BatchNorm, Dropout, L2)
- Recherche exhaustive d’hyperparamètres (Grid Search) et recherche aléatoire (Random Search)
- Évaluation finale et analyse des erreurs (cartes géographiques, résidus)

## Structure du dépôt

| Fichier               | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `ex1_dataset.py`      | Chargement, analyse exploratoire, split train/val/test, standardisation et création des DataLoaders PyTorch. |
| `ex2_model.py`        | Définition de la classe `DeepFFN`, fonctions d’entraînement (`train_one_epoch`, `evaluate`, `train_model`). |
| `ex3_ablation.py`     | Étude d’ablation : suppression de BatchNorm, Dropout ou L2. Analyse du gradient clipping sur données brutes. |
| `ex4_grid_search.py`  | Grid Search sur un sous‑espace réduit (48 configurations). Visualisation de l’impact des hyperparamètres. |
| `ex5_random_search.py`| Random Search avec échantillonnage continu (log‑uniforme). Comparaison avec Grid Search. |
| `ex6_evaluation.py`   | Évaluation finale des meilleures configurations sur le jeu de test. Génération des graphiques (prédit vs réel, résidus, carte des erreurs). |

## Prérequis

- Python 3.8 ou supérieur
- Les bibliothèques suivantes (installables via `pip`) :

```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn scipy
