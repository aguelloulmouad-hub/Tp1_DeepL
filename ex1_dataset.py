# -*- coding: utf-8 -*-
"""
Exercice 1 — Exploration et Preparation du Dataset California Housing
=====================================================================
TP2 Deep Learning — Optimisation des FFN
Universite Sultan Moulay Slimane — FST Beni Mellal
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ── Imports communs ──────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour sauvegarder les figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools, random, time

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =====================================================================
# Partie 1.1 — Chargement et analyse exploratoire
# =====================================================================

# ── Q1 : Chargement du dataset ──────────────────────────────────────
print("=" * 60)
print("Q1 — Chargement et statistiques descriptives")
print("=" * 60)

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print(f"Nombre d'exemples  : {df.shape[0]}")
print(f"Nombre de features : {df.shape[1] - 1}")  # -1 pour la target
print(f"Noms des features  : {housing.feature_names}")
print(f"\n--- 5 premières lignes ---")
print(df.head())
print(f"\n--- Statistiques descriptives ---")
print(df.describe())

# ── Q2 : Distribution de la variable cible ──────────────────────────
print("\n" + "=" * 60)
print("Q2 — Distribution de MedHouseVal")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme
axes[0].hist(df['MedHouseVal'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Valeur médiane des maisons (×100k $)')
axes[0].set_ylabel('Fréquence')
axes[0].set_title('Histogramme de MedHouseVal')
axes[0].axvline(df['MedHouseVal'].mean(), color='red', linestyle='--', label=f"Moyenne = {df['MedHouseVal'].mean():.2f}")
axes[0].axvline(df['MedHouseVal'].median(), color='orange', linestyle='--', label=f"Médiane = {df['MedHouseVal'].median():.2f}")
axes[0].legend()

# Boxplot
axes[1].boxplot(df['MedHouseVal'], vert=True)
axes[1].set_ylabel('Valeur médiane des maisons (×100k $)')
axes[1].set_title('Boxplot de MedHouseVal')

plt.tight_layout()
plt.savefig('q2_distribution_target.png', dpi=120)
plt.show()

# Commentaires
print("""
Commentaire Q2 :
- La distribution est asymétrique à droite (right-skewed).
- Il y a un plafonnement artificiel visible à 5.0 (la valeur max est exactement 5.001),
  ce qui crée un pic anormal en fin de distribution.
- Le boxplot montre des valeurs aberrantes au-dessus de ~4.5.
- La médiane (~1.80) est inférieure à la moyenne (~2.07), confirmant l'asymétrie.
""")

# ── Q3 : Heatmap de corrélation ─────────────────────────────────────
print("=" * 60)
print("Q3 — Heatmap de corrélation")
print("=" * 60)

plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, vmin=-1, vmax=1, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Matrice de corrélation — California Housing')
plt.tight_layout()
plt.savefig('q3_heatmap_correlation.png', dpi=120)
plt.show()

# Analyse des corrélations
target_corr = corr_matrix['MedHouseVal'].drop('MedHouseVal').abs().sort_values(ascending=False)
print(f"\nCorrélation avec la cible (|r|) :")
print(target_corr)
print(f"\n→ Feature la plus corrélée à la cible : {target_corr.index[0]} (r = {corr_matrix.loc[target_corr.index[0], 'MedHouseVal']:.3f})")

# Trouver la plus forte colinéarité entre features
features_only = corr_matrix.drop('MedHouseVal', axis=0).drop('MedHouseVal', axis=1)
feat_vals = features_only.values.copy()
np.fill_diagonal(feat_vals, 0)
max_idx = np.unravel_index(np.abs(feat_vals).argmax(), feat_vals.shape)
f1, f2 = features_only.index[max_idx[0]], features_only.columns[max_idx[1]]
print(f"→ Plus forte colinéarité : {f1} - {f2} (r = {features_only.iloc[max_idx[0], max_idx[1]]:.3f})")

# =====================================================================
# Partie 1.2 — Prétraitement et pipeline PyTorch
# =====================================================================

# ── Q4 : Split 70/15/15 avec stratification ─────────────────────────
print("\n" + "=" * 60)
print("Q4 — Split train/val/test (70/15/15)")
print("=" * 60)

X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values.reshape(-1, 1)

# Création de bins pour la stratification
y_binned = pd.cut(df['MedHouseVal'], bins=5, labels=False)

# Premier split : 70% train / 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y_binned
)

# Recalculer les bins pour le split temp
y_temp_binned = pd.cut(pd.Series(y_temp.ravel()), bins=5, labels=False)

# Deuxième split : 50/50 de temp → 15% val / 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp_binned
)

print(f"Train : {X_train.shape[0]} exemples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Val   : {X_val.shape[0]} exemples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test  : {X_test.shape[0]} exemples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ── Q5 : StandardScaler (fitté sur train uniquement) ────────────────
print("\n" + "=" * 60)
print("Q5 — StandardScaler")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform sur train
X_val_scaled   = scaler.transform(X_val)          # transform seulement sur val
X_test_scaled  = scaler.transform(X_test)         # transform seulement sur test

print(f"Moyenne train après scaling : {X_train_scaled.mean(axis=0).round(4)}")
print(f"Écart-type train après scaling : {X_train_scaled.std(axis=0).round(4)}")

print("""
Pourquoi ne pas fitter sur val/test ?
→ Le StandardScaler doit être fitté uniquement sur les données d'entraînement
  pour éviter le data leakage. Si on utilise les statistiques (mean, std) du
  val/test set pour normaliser, le modèle a accès indirect à des informations
  sur les données qu'il est censé ne jamais avoir vues pendant l'entraînement.
  Cela biaise l'estimation de la performance réelle du modèle.
""")

# ── Q6 : TensorDataset et DataLoader ────────────────────────────────
print("=" * 60)
print("Q6 — Création des DataLoaders PyTorch")
print("=" * 60)

# Conversion en tenseurs PyTorch
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32)

# Création des TensorDataset
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)
test_ds  = TensorDataset(X_test_t, y_test_t)

# Création des DataLoader
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

# Vérification
xb, yb = next(iter(train_loader))
print(f'X batch : {xb.shape}  |  y batch : {yb.shape}')   # [64,8]  [64,1]
print(f'X mean  : {xb.mean():.4f}  |  X std : {xb.std():.4f}')  # ≈0.0, ≈1.0

print("\n✅ Exercice 1 terminé — DataLoaders prêts pour l'entraînement.")
