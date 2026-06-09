# -*- coding: utf-8 -*-
"""
Exercice 4 — Grid Search — Recherche Exhaustive d'Hyperparametres
=================================================================
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

# ── Dataset ──────────────────────────────────────────────────────────
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

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)

# ── Modèle et fonctions d'entraînement ──────────────────────────────
from ex2_model import DeepFFN, train_model


# =====================================================================
# Partie 4.1 — Définition de la grille
# =====================================================================

# ── Q1 : Grille complète + calcul du nombre de configs ──────────────
print("=" * 60)
print("Q1 — Définition de la grille d'hyperparamètres")
print("=" * 60)

param_grid = {
    'hidden_dims'  : [[64, 32], [128, 64], [128, 64, 32], [256, 128, 64]],
    'activation'   : ['relu', 'leaky_relu', 'elu'],
    'dropout_rate' : [0.0, 0.2, 0.3],
    'lr'           : [1e-3, 5e-4],
    'weight_decay' : [0.0, 1e-4],
    'clip_value'   : [1.0, 5.0],
}

n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)
print(f'Total configurations Grid Search : {n_configs}')
# 4 × 3 × 3 × 2 × 2 × 2 = 288


# =====================================================================
# Partie 4.2 — Implémentation du Grid Search
# =====================================================================

# ── Q2 : Fonction grid_search ───────────────────────────────────────
def grid_search(param_grid, train_loader, val_loader, epochs=80):
    """
    Recherche exhaustive (Grid Search) sur toutes les combinaisons
    de la grille d'hyperparamètres.
    Retourne un DataFrame trié par val_mse croissant.
    """
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    results = []

    print(f'Grid Search : {len(combos)} configurations × {epochs} epochs max')
    print('-' * 60)

    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        config['epochs'] = epochs
        config['early_stopping_patience'] = 15
        config['use_bn'] = True

        torch.manual_seed(42)  # reproductibilité
        model = DeepFFN(
            input_dim=8,
            hidden_dims=config['hidden_dims'],
            activation=config['activation'],
            use_bn=config['use_bn'],
            dropout_rate=config['dropout_rate']
        )
        _, best_mse, elapsed = train_model(model, train_loader, val_loader, config)

        # Convertir hidden_dims en string pour le DataFrame
        config_record = {**config}
        config_record['hidden_dims'] = str(config['hidden_dims'])
        config_record['val_mse'] = best_mse
        config_record['time_s'] = elapsed
        results.append(config_record)

        print(f'  [{i+1:3d}/{len(combos)}] val_MSE={best_mse:.4f}  ({elapsed:.1f}s)')

    df_results = pd.DataFrame(results).sort_values('val_mse')
    return df_results


# =====================================================================
# Partie 4.3 — Exécution sur sous-espace réduit
# =====================================================================

# ── Q3 : Grid Search sur 48 configurations ──────────────────────────
print("\n" + "=" * 60)
print("Q3 — Grid Search sur sous-espace réduit (48 configs)")
print("=" * 60)

param_grid_small = {
    'hidden_dims'  : [[64, 32], [128, 64, 32], [256, 128, 64]],
    'activation'   : ['relu', 'leaky_relu'],
    'dropout_rate' : [0.1, 0.3],
    'lr'           : [1e-3, 5e-4],
    'weight_decay' : [1e-4, 1e-3],
    'clip_value'   : [1.0],
}

n_small = 1
for v in param_grid_small.values():
    n_small *= len(v)
print(f'Sous-espace : {n_small} configurations')
# 3 × 2 × 2 × 2 × 2 × 1 = 48

gs_results = grid_search(param_grid_small, train_loader, val_loader, epochs=80)

print('\n=== TOP 10 configurations Grid Search ===')
print(gs_results[['hidden_dims', 'activation', 'dropout_rate',
                   'lr', 'weight_decay', 'val_mse']].head(10).to_string(index=False))

# Sauvegarder les résultats pour comparaison ultérieure
gs_results.to_csv('grid_search_results.csv', index=False)


# ── Q4 : Visualisations des résultats ───────────────────────────────
print("\n" + "=" * 60)
print("Q4 — Visualisations des résultats Grid Search")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Analyse des résultats — Grid Search (48 configurations)', fontsize=14, fontweight='bold')

# a) Boxplot val_mse par activation
sns.boxplot(data=gs_results, x='activation', y='val_mse', ax=axes[0, 0],
            palette='Set2')
axes[0, 0].set_title('Val MSE par activation')
axes[0, 0].grid(alpha=0.3, axis='y')

# b) Boxplot val_mse par dropout_rate
sns.boxplot(data=gs_results, x='dropout_rate', y='val_mse', ax=axes[0, 1],
            palette='Set3')
axes[0, 1].set_title('Val MSE par dropout_rate')
axes[0, 1].grid(alpha=0.3, axis='y')

# c) Heatmap : (lr × weight_decay) → val_mse moyen
pivot = gs_results.pivot_table(values='val_mse', index='lr',
                                columns='weight_decay', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1, 0])
axes[1, 0].set_title('Val MSE moyen : lr × weight_decay')

# d) Barplot des 15 meilleures configurations
top15 = gs_results.head(15).copy()
top15['config_label'] = top15.apply(
    lambda r: f"{r['hidden_dims']}\n{r['activation']}\ndr={r['dropout_rate']}", axis=1)
top15['n_layers'] = top15['hidden_dims'].apply(lambda x: len(eval(x) if isinstance(x, str) else x))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top15)))
bars = axes[1, 1].barh(range(len(top15)), top15['val_mse'].values,
                        color=[colors[int(n)-2] for n in top15['n_layers'].values])
axes[1, 1].set_yticks(range(len(top15)))
axes[1, 1].set_yticklabels([f"#{i+1}" for i in range(len(top15))], fontsize=8)
axes[1, 1].set_xlabel('Val MSE')
axes[1, 1].set_title('Top 15 configurations (couleur = nb couches)')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('q4_grid_search_analysis.png', dpi=120)
plt.show()


# ── Q5 : Impact de chaque hyperparamètre (variance) ─────────────────
print("\n" + "=" * 60)
print("Q5 — Impact de chaque hyperparamètre")
print("=" * 60)

hyperparams = ['hidden_dims', 'activation', 'dropout_rate', 'lr', 'weight_decay']
print(f"\n{'Hyperparamètre':<20} {'Variance val_mse':>18} {'Std val_mse':>15}")
print("-" * 55)

impact_data = []
for hp in hyperparams:
    grouped = gs_results.groupby(hp)['val_mse']
    # Variance des moyennes par groupe → mesure l'impact
    means = grouped.mean()
    variance = means.var()
    std = means.std()
    impact_data.append({'hp': hp, 'variance': variance, 'std': std})
    print(f"{hp:<20} {variance:>18.6f} {std:>15.4f}")

# Identifier le plus impactant
impact_df = pd.DataFrame(impact_data).sort_values('variance', ascending=False)
print(f"\n→ Hyperparamètre le plus influent : {impact_df.iloc[0]['hp']}")
print(f"  (variance des moyennes par groupe = {impact_df.iloc[0]['variance']:.6f})")

print("""
Justification :
L'hyperparamètre avec la plus grande variance des moyennes de val_mse
entre ses différentes valeurs est celui qui a le plus grand impact sur
la performance. Cela signifie que changer sa valeur modifie significativement
le résultat, indépendamment des autres hyperparamètres.
""")

print("\n✅ Exercice 4 terminé — Grid Search exécuté et analysé.")
