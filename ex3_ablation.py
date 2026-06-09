# -*- coding: utf-8 -*-
"""
Exercice 3 — Analyse de l'Impact de Chaque Technique de Stabilisation
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools, random, time

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Réutilisation du code des exercices précédents ──────────────────
# --- Dataset ---
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target
X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values.reshape(-1, 1)
y_binned = pd.cut(df['MedHouseVal'], bins=5, labels=False)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y_binned)
y_temp_binned = pd.cut(pd.Series(y_temp.ravel()), bins=5, labels=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp_binned)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)

# --- Modèle et fonctions d'entraînement (depuis ex2_model.py) ---
from ex2_model import DeepFFN, train_one_epoch, evaluate, train_model


# =====================================================================
# Partie 3.1 — Ablation study : BatchNorm vs Dropout vs L2
# =====================================================================

# ── Q1 : 5 variantes (A-E) ──────────────────────────────────────────
print("=" * 60)
print("Q1 — Ablation study : 5 configurations")
print("=" * 60)

ablation_configs = {
    'A — Baseline complet': {
        'hidden_dims': [128, 64, 32], 'activation': 'relu',
        'use_bn': True, 'dropout_rate': 0.2,
        'lr': 1e-3, 'weight_decay': 1e-4,
        'clip_value': 1.0, 'epochs': 100, 'early_stopping_patience': 15,
    },
    'B — Sans BatchNorm': {
        'hidden_dims': [128, 64, 32], 'activation': 'relu',
        'use_bn': False, 'dropout_rate': 0.2,
        'lr': 1e-3, 'weight_decay': 1e-4,
        'clip_value': 1.0, 'epochs': 100, 'early_stopping_patience': 15,
    },
    'C — Sans Dropout': {
        'hidden_dims': [128, 64, 32], 'activation': 'relu',
        'use_bn': True, 'dropout_rate': 0.0,
        'lr': 1e-3, 'weight_decay': 1e-4,
        'clip_value': 1.0, 'epochs': 100, 'early_stopping_patience': 15,
    },
    'D — Sans L2 (wd=0)': {
        'hidden_dims': [128, 64, 32], 'activation': 'relu',
        'use_bn': True, 'dropout_rate': 0.2,
        'lr': 1e-3, 'weight_decay': 0.0,
        'clip_value': 1.0, 'epochs': 100, 'early_stopping_patience': 15,
    },
    'E — Aucune régul.': {
        'hidden_dims': [128, 64, 32], 'activation': 'relu',
        'use_bn': False, 'dropout_rate': 0.0,
        'lr': 1e-3, 'weight_decay': 0.0,
        'clip_value': 1.0, 'epochs': 100, 'early_stopping_patience': 15,
    },
}

ablation_results = {}
ablation_histories = {}

for name, config in ablation_configs.items():
    print(f'\n--- {name} ---')
    torch.manual_seed(42)
    model = DeepFFN(
        hidden_dims=config['hidden_dims'],
        activation=config['activation'],
        use_bn=config['use_bn'],
        dropout_rate=config['dropout_rate']
    )
    history, best_mse, elapsed = train_model(model, train_loader, val_loader, config)
    ablation_results[name] = {'best_val_mse': best_mse, 'time_s': elapsed}
    ablation_histories[name] = history
    print(f'  → best_val_MSE = {best_mse:.4f}  |  temps = {elapsed:.1f}s')

# Tableau récapitulatif
print("\n" + "=" * 60)
print("Résultats de l'ablation study :")
print("=" * 60)
print(f"{'Configuration':<30} {'Val MSE':>10} {'Temps (s)':>10}")
print("-" * 52)
for name, res in ablation_results.items():
    print(f"{name:<30} {res['best_val_mse']:>10.4f} {res['time_s']:>10.1f}")


# ── Q2 : Courbes comparatives val_mse ───────────────────────────────
print("\n" + "=" * 60)
print("Q2 — Courbes comparatives val_mse")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Ablation Study — Comparaison des courbes val_MSE', fontsize=14, fontweight='bold')

# A vs B (BatchNorm)
comparisons = [
    ('A vs B — Effet BatchNorm', 'A — Baseline complet', 'B — Sans BatchNorm'),
    ('A vs C — Effet Dropout',   'A — Baseline complet', 'C — Sans Dropout'),
    ('A vs E — Effet régul.',    'A — Baseline complet', 'E — Aucune régul.'),
]

for ax, (title, name1, name2) in zip(axes, comparisons):
    h1 = ablation_histories[name1]
    h2 = ablation_histories[name2]
    ax.plot(h1['val_mse'], label=name1, linewidth=1.5)
    ax.plot(h2['val_mse'], label=name2, linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val MSE')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('q2_ablation_comparison.png', dpi=120)
plt.show()

print("""
Commentaire Q2 :
- BatchNorm (A vs B) : apporte la plus grande stabilité — la convergence est
  plus rapide et plus lisse avec BatchNorm. Sans BN, les courbes sont plus bruitées.
- Dropout (A vs C) : prévient le surapprentissage — sans Dropout, l'écart
  train/val tend à se creuser au fil des epochs.
- Toutes les régul. (A vs E) : sans aucune régularisation, le modèle surapprend
  fortement, la val_MSE remonte après quelques epochs.
""")


# =====================================================================
# Partie 3.2 — Effet du Gradient Clipping
# =====================================================================

# ── Q3 : Effet du clip_value ────────────────────────────────────────
print("=" * 60)
print("Q3 — Effet du gradient clipping")
print("=" * 60)

clip_values = [None, 0.1, 0.5, 1.0, 5.0, 10.0]
clip_histories = {}

for cv in clip_values:
    label = f'clip={cv}' if cv is not None else 'sans clipping'
    print(f'\n--- {label} ---')
    torch.manual_seed(42)
    model = DeepFFN(hidden_dims=[128, 64, 32], activation='relu',
                    use_bn=True, dropout_rate=0.2)
    
    config_clip = {
        'lr': 1e-3, 'weight_decay': 1e-4,
        'clip_value': cv if cv is not None else float('inf'),
        'epochs': 50, 'early_stopping_patience': 50,  # pas d'early stopping
    }
    history, best_mse, _ = train_model(model, train_loader, val_loader, config_clip)
    clip_histories[label] = history
    print(f'  → best_val_MSE = {best_mse:.4f}')

# Tracé des courbes de gradient_norm
plt.figure(figsize=(12, 6))
for label, hist in clip_histories.items():
    plt.plot(hist['grad_norm'], label=label, alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Norme du gradient par epoch — différents clip_value')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('q3_gradient_clipping.png', dpi=120)
plt.show()


# ── Q4 : Distribution des normes de gradient (boxplot) ──────────────
print("\n" + "=" * 60)
print("Q4 — Distribution des normes de gradient")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))
data_for_box = []
labels_for_box = []
for label, hist in clip_histories.items():
    data_for_box.append(hist['grad_norm'])
    labels_for_box.append(label)

ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7))
ax.set_ylabel('Gradient Norm')
ax.set_title('Distribution des normes de gradient — par clip_value')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('q4_gradient_boxplot.png', dpi=120)
plt.show()

print("""
Commentaire Q4 :
- clip=0.1 : bride excessivement les gradients → convergence très lente (underfitting)
- clip=0.5 / 1.0 : bon compromis — les gradients restent contrôlés sans brider l'apprentissage
- clip=5.0 / 10.0 : peu d'effet de clipping, se rapproche du comportement sans clipping
- La valeur clip=1.0 offre généralement le meilleur compromis.
""")


# ── Q5 : Effet du clipping sur données non normalisées ──────────────
print("=" * 60)
print("Q5 — Données brutes (sans StandardScaler) + clipping")
print("=" * 60)

# Créer des DataLoaders avec données brutes (non normalisées)
X_train_raw_t = torch.tensor(X_train, dtype=torch.float32)
X_val_raw_t   = torch.tensor(X_val, dtype=torch.float32)
train_loader_raw = DataLoader(TensorDataset(X_train_raw_t, y_train_t), batch_size=64, shuffle=True)
val_loader_raw   = DataLoader(TensorDataset(X_val_raw_t, y_val_t), batch_size=256, shuffle=False)

raw_histories = {}
for cv_label, cv in [('Brut + sans clipping', float('inf')), ('Brut + clip=1.0', 1.0)]:
    print(f'\n--- {cv_label} ---')
    torch.manual_seed(42)
    model = DeepFFN(hidden_dims=[128, 64, 32], activation='relu',
                    use_bn=True, dropout_rate=0.2)
    config_raw = {
        'lr': 1e-3, 'weight_decay': 1e-4,
        'clip_value': cv, 'epochs': 50, 'early_stopping_patience': 50,
    }
    history, best_mse, _ = train_model(model, train_loader_raw, val_loader_raw, config_raw)
    raw_histories[cv_label] = history
    print(f'  → best_val_MSE = {best_mse:.4f}')

# Comparaison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Effet du clipping sur données brutes (sans StandardScaler)', fontsize=13, fontweight='bold')

# Val MSE
for label, hist in raw_histories.items():
    axes[0].plot(hist['val_mse'], label=label, linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Val MSE')
axes[0].set_title('Val MSE — données brutes')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Gradient Norm
for label, hist in raw_histories.items():
    axes[1].plot(hist['grad_norm'], label=label, linewidth=1.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Gradient Norm')
axes[1].set_title('Gradient Norm — données brutes')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('q5_raw_data_clipping.png', dpi=120)
plt.show()

print("""
Commentaire Q5 :
- Sans StandardScaler, les features ont des échelles très différentes
  (ex: Population en milliers vs Latitude en degrés), ce qui provoque
  des gradients très instables et potentiellement explosifs.
- Sans clipping, les normes de gradient sont très élevées et erratiques.
- Avec clipping (clip=1.0), les gradients sont contrôlés et l'entraînement
  est stabilisé, même sur données brutes. Le clipping agit comme un filet
  de sécurité.
- Néanmoins, la normalisation des données reste préférable car elle permet
  une meilleure convergence et des performances supérieures.
""")

print("\n✅ Exercice 3 terminé — Ablation study et analyse du gradient clipping.")
