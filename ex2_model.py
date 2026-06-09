# -*- coding: utf-8 -*-
"""
Exercice 2 — Implementation du MLP avec Regularisation integree
================================================================
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

# ── Réutilisation du dataset préparé dans l'Exercice 1 ──────────────
# Chargement et préparation (identique à ex1_dataset.py)
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values.reshape(-1, 1)
y_binned = pd.cut(df['MedHouseVal'], bins=5, labels=False)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y_binned)
y_temp_binned = pd.cut(pd.Series(y_temp.ravel()), bins=5, labels=False)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp_binned)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=256, shuffle=False)


# =====================================================================
# Partie 2.1 — Architecture modulaire
# =====================================================================

# ── Q1 : Classe DeepFFN ─────────────────────────────────────────────
class DeepFFN(nn.Module):
    """
    Réseau feedforward profond avec :
      - Architecture configurable (hidden_dims)
      - Activation configurable ('relu','tanh','leaky_relu','elu','selu')
      - BatchNorm1d optionnelle (use_bn)
      - Dropout configurable (dropout_rate, 0.0 = désactivé)
      - Initialisation He (ReLU) ou Xavier (tanh/autres)
    """
    def __init__(self,
                 input_dim   : int   = 8,
                 hidden_dims : list  = None,
                 output_dim  : int   = 1,
                 activation  : str   = 'relu',
                 use_bn      : bool  = True,
                 dropout_rate: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.activation_name = activation
        self.layers = nn.ModuleList()

        # Construction des couches cachées
        prev_dim = input_dim
        for h_dim in hidden_dims:
            block = nn.ModuleList()
            block.append(nn.Linear(prev_dim, h_dim))
            if use_bn:
                block.append(nn.BatchNorm1d(h_dim))
            block.append(self._get_activation(activation))
            if dropout_rate > 0.0:
                block.append(nn.Dropout(dropout_rate))
            self.layers.append(block)
            prev_dim = h_dim

        # Couche de sortie (régression → pas d'activation)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Initialisation des poids
        self._init_weights()

    def _get_activation(self, name):
        """Retourne le module d'activation correspondant au nom."""
        activations = {
            'relu'       : nn.ReLU(),
            'tanh'       : nn.Tanh(),
            'leaky_relu' : nn.LeakyReLU(0.01),
            'elu'        : nn.ELU(),
            'selu'       : nn.SELU(),
        }
        if name not in activations:
            raise ValueError(f"Activation '{name}' non supportée. "
                             f"Choix : {list(activations.keys())}")
        return activations[name]

    # ── Q2 : Initialisation des poids ────────────────────────────────
    def _init_weights(self):
        """
        He init pour relu/leaky_relu/elu
        Xavier init pour tanh/selu/autres
        Biais initialisés à zéro.
        """
        he_activations = {'relu', 'leaky_relu', 'elu'}
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_name in he_activations:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Propagation à travers les couches cachées puis la couche de sortie."""
        for block in self.layers:
            for layer in block:
                x = layer(x)
        x = self.output_layer(x)
        return x


# =====================================================================
# Partie 2.2 — Gradient Clipping et boucle d'entraînement
# =====================================================================

# ── Q4 : train_one_epoch ────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, clip_value=1.0):
    """
    Entraine le modele pour une epoch avec gradient clipping.
    Retourne : (perte moyenne, norme moyenne du gradient)
    """
    model.train()
    total_loss, total_gnorm, n = 0.0, 0.0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()

        # Gradient clipping
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        total_gnorm += gnorm.item()

        optimizer.step()
        total_loss += loss.item() * len(xb)
        n += len(xb)
    return total_loss / n, total_gnorm / len(loader)


# ── Q5 : evaluate ───────────────────────────────────────────────────
def evaluate(model, loader, criterion):
    """
    Evalue le modele sur un DataLoader.
    Retourne : (MSE, MAE, R2)
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            all_preds.append(pred.numpy())
            all_targets.append(yb.numpy())

    all_preds   = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2  = r2_score(all_targets, all_preds)

    return mse, mae, r2


# ── Q6 : train_model (boucle complete) ──────────────────────────────
def train_model(model, train_loader, val_loader, config):
    """
    Boucle d'entrainement complete avec :
      - Adam optimizer (lr + weight_decay)
      - ReduceLROnPlateau scheduler
      - Early stopping
      - Sauvegarde du meilleur modele
    Retourne : (history, best_val_mse, temps_entrainement)
    """
    optimizer = optim.Adam(model.parameters(),
                           lr=config['lr'],
                           weight_decay=config.get('weight_decay', 0.0))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    clip_value = config.get('clip_value', 1.0)

    history = {'train_mse': [], 'val_mse': [], 'val_mae': [],
               'val_r2': [], 'lr': [], 'grad_norm': []}
    best_val = float('inf')
    patience = config.get('early_stopping_patience', 20)
    no_improve = 0
    t0 = time.time()

    for epoch in range(config.get('epochs', 200)):
        tr_loss, gnorm = train_one_epoch(model, train_loader, optimizer,
                                         criterion, clip_value)
        va_mse, va_mae, va_r2 = evaluate(model, val_loader, criterion)
        scheduler.step(va_mse)

        history['train_mse'].append(tr_loss)
        history['val_mse'].append(va_mse)
        history['val_mae'].append(va_mae)
        history['val_r2'].append(va_r2)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['grad_norm'].append(gnorm)

        if va_mse < best_val:
            best_val = va_mse
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping epoch {epoch+1}')
                break

        if (epoch + 1) % 20 == 0:
            print(f'Ep {epoch+1:3d} | tr_MSE={tr_loss:.4f} | '
                  f'val_MSE={va_mse:.4f} | R2={va_r2:.4f} | '
                  f'lr={optimizer.param_groups[0]["lr"]:.6f}')

    return history, best_val, time.time() - t0


if __name__ == '__main__':
    # ── Q3 : Test du modèle ─────────────────────────────────────────────
    print("=" * 60)
    print("Q3 — Architecture et paramètres du modèle")
    print("=" * 60)

    torch.manual_seed(42)
    model = DeepFFN()
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nParamètres entraînables : {total_params:,}')

    # Test forward pass
    xb, _ = next(iter(train_loader))
    out = model(xb)
    print(f'Sortie : {out.shape}')   # [64, 1]
    print(f'Plage  : [{out.min().item():.3f}, {out.max().item():.3f}]')

    # ── Q7 : Entraînement baseline + visualisation ──────────────────────
    print("\n" + "=" * 60)
    print("Q7 — Entraînement du modèle baseline")
    print("=" * 60)

    config_baseline = {
        'hidden_dims'  : [128, 64, 32],
        'activation'   : 'relu',
        'use_bn'       : True,
        'dropout_rate' : 0.2,
        'lr'           : 1e-3,
        'weight_decay' : 1e-4,
        'clip_value'   : 1.0,
        'epochs'       : 200,
        'early_stopping_patience': 25,
    }

    torch.manual_seed(42)
    model_baseline = DeepFFN(
        hidden_dims=config_baseline['hidden_dims'],
        activation=config_baseline['activation'],
        use_bn=config_baseline['use_bn'],
        dropout_rate=config_baseline['dropout_rate']
    )

    history_baseline, best_val_baseline, time_baseline = train_model(
        model_baseline, train_loader, val_loader, config_baseline
    )

    print(f'\n-> Meilleur val_MSE baseline : {best_val_baseline:.4f}')
    print(f'-> Temps d\'entraînement     : {time_baseline:.1f}s')

    # ── Visualisation des courbes d'apprentissage (5 sous-graphes) ──────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Courbes d\'apprentissage — Modele Baseline', fontsize=14, fontweight='bold')
    epochs_range = range(1, len(history_baseline['train_mse']) + 1)

    # 1) Train MSE vs Val MSE
    axes[0, 0].plot(epochs_range, history_baseline['train_mse'], label='Train MSE', color='blue')
    axes[0, 0].plot(epochs_range, history_baseline['val_mse'], label='Val MSE', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Train MSE vs Val MSE')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2) Val MSE seul (zoom)
    axes[0, 1].plot(epochs_range, history_baseline['val_mse'], color='red', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Val MSE')
    axes[0, 1].set_title('Validation MSE')
    axes[0, 1].grid(alpha=0.3)

    # 3) Val R²
    axes[0, 2].plot(epochs_range, history_baseline['val_r2'], color='green', linewidth=1.5)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('R2')
    axes[0, 2].set_title('Validation R2')
    axes[0, 2].grid(alpha=0.3)

    # 4) Learning Rate
    axes[1, 0].plot(epochs_range, history_baseline['lr'], color='purple', linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate (ReduceLROnPlateau)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)

    # 5) Gradient Norm
    axes[1, 1].plot(epochs_range, history_baseline['grad_norm'], color='orange', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gradient Norm')
    axes[1, 1].set_title('Norme du Gradient (apres clipping)')
    axes[1, 1].grid(alpha=0.3)

    # Cacher le 6e subplot inutilisé
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('q7_baseline_curves.png', dpi=120)
    plt.show()

    print("\n[OK] Exercice 2 termine — Modele baseline entraine et visualise.")

