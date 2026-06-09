# -*- coding: utf-8 -*-
"""
Exercice 5 — Random Search — Exploration Efficace de l'Espace Continu
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
from scipy.stats import spearmanr
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

# ── Modèle et fonctions ─────────────────────────────────────────────
from ex2_model import DeepFFN, train_model


# =====================================================================
# Partie 5.1 — Espace de recherche continu
# =====================================================================

# ── Q1 : Espace de recherche + sample_config ────────────────────────
print("=" * 60)
print("Q1 — Espace de recherche continu + sample_config")
print("=" * 60)

search_space = {
    # Distributions continues (log-uniform pour lr et wd)
    'lr'           : ('log_uniform', 1e-4, 1e-2),
    'weight_decay' : ('log_uniform', 1e-5, 1e-2),
    'dropout_rate' : ('uniform',     0.0,  0.5),
    'clip_value'   : ('uniform',     0.5,  5.0),

    # Distributions discrètes
    'hidden_dims'  : ('choice', [[64, 32], [128, 64], [128, 64, 32],
                                  [256, 128, 64], [256, 128, 64, 32]]),
    'activation'   : ('choice', ['relu', 'leaky_relu', 'elu', 'selu']),
}


def sample_config(space):
    """
    Tire une configuration aléatoire depuis l'espace de recherche.
    Supporte : log_uniform, uniform, choice.
    """
    config = {}
    for key, spec in space.items():
        dist = spec[0]
        if dist == 'log_uniform':
            # Échantillonnage log-uniforme : exp(U[log(min), log(max)])
            val = np.exp(np.random.uniform(np.log(spec[1]), np.log(spec[2])))
            config[key] = val
        elif dist == 'uniform':
            config[key] = random.uniform(spec[1], spec[2])
        elif dist == 'choice':
            config[key] = random.choice(spec[1])
    return config


# Test
print("Exemples de configurations tirées aléatoirement :")
for i in range(3):
    cfg = sample_config(search_space)
    print(f"  Config {i+1}: lr={cfg['lr']:.2e}, wd={cfg['weight_decay']:.2e}, "
          f"dr={cfg['dropout_rate']:.2f}, clip={cfg['clip_value']:.2f}, "
          f"act={cfg['activation']}, dims={cfg['hidden_dims']}")


# =====================================================================
# Partie 5.2 — Implémentation du Random Search
# =====================================================================

# ── Q2 : Fonction random_search ─────────────────────────────────────
def random_search(search_space, n_trials, train_loader, val_loader, epochs=80):
    """
    Recherche aléatoire (Random Search) avec n_trials tirages.
    Retourne un DataFrame trié par val_mse croissant.
    """
    results = []
    print(f'Random Search : {n_trials} tirages × {epochs} epochs max')
    print('-' * 60)

    for trial in range(n_trials):
        config = sample_config(search_space)
        config['epochs'] = epochs
        config['early_stopping_patience'] = 15
        config['use_bn'] = True

        torch.manual_seed(trial)  # seed différente par trial !
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
        config_record['trial'] = trial
        config_record['time_s'] = elapsed
        results.append(config_record)

        print(f'  Trial {trial+1:3d}/{n_trials} | val_MSE={best_mse:.4f}  ({elapsed:.1f}s)')
        print(f'  lr={config["lr"]:.2e}  wd={config["weight_decay"]:.2e}  '
              f'dr={config["dropout_rate"]:.2f}  act={config["activation"]}')

    df_results = pd.DataFrame(results).sort_values('val_mse')
    return df_results


# ── Q3 : Exécution (48 trials) ──────────────────────────────────────
print("\n" + "=" * 60)
print("Q3 — Exécution du Random Search (48 trials)")
print("=" * 60)

# Fixer la seed pour la reproductibilité du tirage
np.random.seed(42)
random.seed(42)

rs_results = random_search(search_space, n_trials=48, train_loader=train_loader,
                           val_loader=val_loader, epochs=80)

print('\n=== TOP 10 configurations Random Search ===')
print(rs_results[['hidden_dims', 'activation', 'dropout_rate',
                   'lr', 'weight_decay', 'val_mse']].head(10).to_string(index=False))

# Sauvegarder les résultats
rs_results.to_csv('random_search_results.csv', index=False)


# =====================================================================
# Partie 5.3 — Comparaison Grid Search vs Random Search
# =====================================================================

# ── Q4 : Courbe de convergence (best-so-far) ────────────────────────
print("\n" + "=" * 60)
print("Q4 — Comparaison Grid Search vs Random Search")
print("=" * 60)


def best_so_far(df_results):
    """Retourne le meilleur MSE cumulatif après chaque essai."""
    # Utiliser l'ordre original d'évaluation (pas trié)
    if 'trial' in df_results.columns:
        ordered = df_results.sort_values('trial')['val_mse'].values
    else:
        ordered = df_results['val_mse'].values
    return np.minimum.accumulate(ordered)


# Charger les résultats du Grid Search
try:
    gs_results = pd.read_csv('grid_search_results.csv')
except FileNotFoundError:
    print("⚠ Résultats Grid Search non trouvés. Exécutez d'abord ex4_grid_search.py")
    gs_results = None

plt.figure(figsize=(10, 6))
if gs_results is not None:
    bsf_gs = best_so_far(gs_results)
    plt.plot(range(1, len(bsf_gs) + 1), bsf_gs, label='Grid Search',
             linewidth=2, color='blue', marker='o', markersize=3)

bsf_rs = best_so_far(rs_results)
plt.plot(range(1, len(bsf_rs) + 1), bsf_rs, label='Random Search',
         linewidth=2, color='red', marker='s', markersize=3)

plt.xlabel('Nombre de configurations évaluées')
plt.ylabel('Meilleur val_MSE trouvé')
plt.title('Convergence : Grid Search vs Random Search')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('gs_vs_rs.png', dpi=120)
plt.show()

print("""
Commentaire Q4 :
- Le Random Search converge généralement plus rapidement vers de bonnes
  performances car il explore l'espace de manière plus diversifiée.
- Le Grid Search est limité à une grille prédéfinie et peut rater des
  zones intéressantes entre les points de la grille.
- Avec le même budget (48 configurations), le Random Search explore
  effectivement plus de régions de l'espace des hyperparamètres.
""")


# ── Q5 : Analyse des hyperparamètres (scatter plots) ────────────────
print("=" * 60)
print("Q5 — Analyse des distributions des hyperparamètres")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Analyse des hyperparamètres — Random Search', fontsize=14, fontweight='bold')

# a) lr vs val_mse (échelle log), coloré par dropout_rate
scatter = axes[0].scatter(rs_results['lr'], rs_results['val_mse'],
                          c=rs_results['dropout_rate'], cmap='viridis',
                          s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
axes[0].set_xscale('log')
axes[0].set_xlabel('Learning Rate (log)')
axes[0].set_ylabel('Val MSE')
axes[0].set_title('lr vs val_MSE (couleur = dropout_rate)')
plt.colorbar(scatter, ax=axes[0], label='dropout_rate')
axes[0].grid(alpha=0.3)

# b) weight_decay vs val_mse (échelle log)
axes[1].scatter(rs_results['weight_decay'], rs_results['val_mse'],
                c='steelblue', s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
axes[1].set_xscale('log')
axes[1].set_xlabel('Weight Decay (log)')
axes[1].set_ylabel('Val MSE')
axes[1].set_title('weight_decay vs val_MSE')
axes[1].grid(alpha=0.3)

# c) Violin plot : dropout_rate pour top 10 vs bottom 10
top10 = rs_results.head(10)
bottom10 = rs_results.tail(10)
violin_data = pd.DataFrame({
    'dropout_rate': pd.concat([top10['dropout_rate'], bottom10['dropout_rate']]),
    'Groupe': ['Top 10'] * len(top10) + ['Pire 10'] * len(bottom10)
})
sns.violinplot(data=violin_data, x='Groupe', y='dropout_rate', ax=axes[2],
               palette=['green', 'red'], inner='box')
axes[2].set_title('Dropout Rate : Top 10 vs Pire 10')
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('q5_rs_hyperparams_analysis.png', dpi=120)
plt.show()


# ── Q6 : Corrélation de Spearman ────────────────────────────────────
print("\n" + "=" * 60)
print("Q6 — Corrélation de Spearman avec val_mse")
print("=" * 60)

continuous_hps = ['lr', 'weight_decay', 'dropout_rate', 'clip_value']

print(f"\n{'Hyperparamètre':<20} {'Spearman ρ':>12} {'p-value':>12} {'Interprétation':>20}")
print("-" * 66)

spearman_results = []
for hp in continuous_hps:
    rho, pval = spearmanr(rs_results[hp], rs_results['val_mse'])
    interpretation = "Fort" if abs(rho) > 0.5 else ("Modéré" if abs(rho) > 0.3 else "Faible")
    spearman_results.append({'hp': hp, 'rho': rho, 'pval': pval})
    print(f"{hp:<20} {rho:>12.4f} {pval:>12.4e} {interpretation:>20}")

# Identifier le plus influent
sp_df = pd.DataFrame(spearman_results)
sp_df['abs_rho'] = sp_df['rho'].abs()
most_influential = sp_df.loc[sp_df['abs_rho'].idxmax()]

print(f"\n→ Hyperparamètre le plus influent : {most_influential['hp']}")
print(f"  (Spearman ρ = {most_influential['rho']:.4f}, p = {most_influential['pval']:.4e})")

print("""
Tableau de synthèse :
La corrélation de Spearman mesure la relation monotone (pas nécessairement
linéaire) entre chaque hyperparamètre et la val_mse. Un ρ positif signifie
qu'augmenter l'hyperparamètre tend à augmenter la MSE (= dégrader les perfs).
Un ρ négatif indique l'inverse. La p-value indique la significativité
statistique de la corrélation.
""")

print("\n✅ Exercice 5 terminé — Random Search exécuté et comparé au Grid Search.")
