import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from mamba_ssm import Mamba

# ── Reproducibility ───────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)

# ── Load Nile data ────────────────────────────────────────────────────
nile   = sm.datasets.nile.load_pandas().data
series = torch.tensor(nile['volume'].values, dtype=torch.float32)

print(f"Series length: {len(series)}")
print(f"Mean:          {series.mean():.4f}")
print(f"Std:           {series.std():.4f}")
print(f"Min:           {series.min():.4f}")
print(f"Max:           {series.max():.4f}")

# ── ADF test on unnormalized series ───────────────────────────────────
result = adfuller(series.numpy())
print(f"\nADF test (original scale):")
print(f"ADF statistic: {result[0]:.4f}")
print(f"p-value:       {result[1]:.4f}")
print(f"{'Stationary' if result[1] < 0.05 else 'Non-stationary'}")

# ── Plot unnormalized full series ─────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(series.numpy(), color='blue', alpha=0.7)
plt.title('Nile River Annual Flow Volume (Original Scale)')
plt.xlabel('Year')
plt.ylabel('Volume')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('nile_full_series_raw.png', dpi=150, bbox_inches='tight')
plt.close()
print("Unnormalized series plot saved to nile_full_series_raw.png")

# ── ACF and PACF on unnormalized series ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series.numpy(),  lags=40, ax=axes[0],
         title='Nile River ACF (Original Scale)')
plot_pacf(series.numpy(), lags=40, ax=axes[1],
         title='Nile River PACF (Original Scale)')
plt.tight_layout()
plt.savefig('nile_acf_pacf_raw.png', dpi=150, bbox_inches='tight')
plt.close()
print("Unnormalized ACF/PACF saved to nile_acf_pacf_raw.png")

# ── Normalize ─────────────────────────────────────────────────────────
mean_val = series.mean().item()
std_val  = series.std().item()
series   = (series - mean_val) / std_val

print(f"\nAfter normalization:")
print(f"Mean: {series.mean():.4f}")
print(f"Std:  {series.std():.4f}")

# ── ADF test on normalized series ────────────────────────────────────
result_norm = adfuller(series.numpy())
print(f"\nADF test (normalized):")
print(f"ADF statistic: {result_norm[0]:.4f}")
print(f"p-value:       {result_norm[1]:.4f}")
print(f"{'Stationary' if result_norm[1] < 0.05 else 'Non-stationary'}")

# ── Plot normalized full series ───────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(series.numpy(), color='blue', alpha=0.7)
plt.title('Nile River Annual Flow Volume (Normalized)')
plt.xlabel('Year')
plt.ylabel('Volume (normalized)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('nile_full_series_norm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Normalized series plot saved to nile_full_series_norm.png")

# ── ACF and PACF on normalized series ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series.numpy(),  lags=40, ax=axes[0],
         title='Nile River ACF (Normalized)')
plot_pacf(series.numpy(), lags=40, ax=axes[1],
         title='Nile River PACF (Normalized)')
plt.tight_layout()
plt.savefig('nile_acf_pacf_norm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Normalized ACF/PACF saved to nile_acf_pacf_norm.png")

# ── Hyperparameters ───────────────────────────────────────────────────
d_model   = 16
d_state   = 16
d_inner   = 32
d_conv    = 4
dt_rank   = math.ceil(d_model / 16)
n_epochs  = 60
lr        = 1e-3
n_samples = 1

# ── Train test split ──────────────────────────────────────────────────
T_total    = len(series)
T_train    = int(T_total * 0.9)
T_test     = T_total - T_train
train_data = series[:T_train].unsqueeze(0)
test_data  = series[T_train:].unsqueeze(0)

print(f"\nT_total : {T_total}")
print(f"T_train : {T_train}")
print(f"T_test  : {T_test}")

# ── Sigma estimate on normalized series ───────────────────────────────
diffs = series[1:] - series[:-1]
sigma = diffs.std().item() / math.sqrt(2)
print(f"sigma   : {sigma:.6f}")
print(f"sigma^2 : {sigma**2:.6f}")

# ── Device ────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# ── Training ──────────────────────────────────────────────────────────
print("\nFitting Mamba to Nile training data...")
all_sample_mse = []

for s in range(n_samples):

    embed  = nn.Linear(1, d_model).to(device)
    mamba  = Mamba(d_model=d_model, d_state=d_state,
                   d_conv=d_conv, expand=2).to(device)
    head   = nn.Linear(d_model, 1).to(device)

    all_params = list(embed.parameters()) + \
                 list(mamba.parameters()) + \
                 list(head.parameters())

    optimizer = torch.optim.Adam(all_params, lr=lr)

    seq   = train_data[s].unsqueeze(0).to(device)
    x_in  = seq[:, :-1].float()
    x_tgt = seq[:, 1:].float()

    for epoch in range(n_epochs):
        embed.train(); mamba.train(); head.train()
        optimizer.zero_grad()
        x_emb = embed(x_in.unsqueeze(-1))
        x_out = mamba(x_emb)
        x_hat = head(x_out).squeeze(-1)
        loss  = F.mse_loss(x_hat, x_tgt)
        loss.backward()
        optimizer.step()

        if epoch % 250 == 0:
            print(f"Epoch {epoch:>4d} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Ratio: {loss.item()/sigma**2:.4f}")

        if loss.item() / sigma**2 <= 1.01:
            print(f"Early stop at epoch {epoch} | "
                  f"Ratio: {loss.item()/sigma**2:.4f}")
            break

    # ── Training MSE ──────────────────────────────────────────────
    embed.eval(); mamba.eval(); head.eval()
    with torch.no_grad():
        x_emb      = embed(x_in.unsqueeze(-1))
        x_out      = mamba(x_emb)
        x_hat      = head(x_out).squeeze(-1)
        errors     = (x_hat - x_tgt) ** 2
        sample_mse = errors.mean().item()

    all_sample_mse.append(sample_mse)
    train_mse_raw = sample_mse * std_val**2
    print(f"\nTrain MSE (normalized)     : {sample_mse:.6f} | "
          f"Ratio: {sample_mse/sigma**2:.4f}")
    print(f"Train MSE (original scale) : {train_mse_raw:.4f}")

    # ── Training fit plot ─────────────────────────────────────────
    x_hat_raw = x_hat.cpu().squeeze().numpy() * std_val + mean_val
    x_tgt_raw = x_tgt.cpu().squeeze().numpy() * std_val + mean_val

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(x_tgt_raw, label='True', color='blue', alpha=0.7)
    axes[0].plot(x_hat_raw, label='Predicted', color='orange', alpha=0.7)
    axes[0].set_title('Mamba Fit — Nile River Full Training Sequence')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Volume')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_tgt_raw[-50:], label='True', color='blue', alpha=0.7)
    axes[1].plot(x_hat_raw[-50:], label='Predicted', color='orange', alpha=0.7)
    axes[1].set_title('Mamba Fit — Last 50 Timesteps')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Volume')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mamba_nile_fit.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training fit plot saved to mamba_nile_fit.png")

    # ── Test set evaluation ───────────────────────────────────────
    print("\nRunning test set evaluation...")
    sample_test_errors = []
    predictions        = []
    true_values        = []

    with torch.no_grad():
        # warm up hidden state on full training sequence
        x_warmup = series[:T_train].unsqueeze(0).to(device).float()
        x_emb    = embed(x_warmup.unsqueeze(-1))
        x_out    = mamba(x_emb)

        # rolling forecast on test set
        for t in range(T_test - 1):
            x_input = series[T_train + t - 1].unsqueeze(0).unsqueeze(0).to(device).float()
            x_emb   = embed(x_input.unsqueeze(-1))
            x_out   = mamba(x_emb)
            x_pred  = head(x_out).squeeze()
            x_true  = series[T_train + t].to(device).float()
            error   = (x_pred - x_true) ** 2

            sample_test_errors.append(error.item())
            predictions.append(x_pred.item())
            true_values.append(x_true.item())

            pred_raw = x_pred.item() * std_val + mean_val
            true_raw = x_true.item() * std_val + mean_val
            print(f"Step {t+1:>3d}/{T_test-1} | "
                  f"Pred: {pred_raw:.2f} | "
                  f"True: {true_raw:.2f} | "
                  f"SE (norm): {error.item():.6f}")

    sample_test_mse     = np.mean(sample_test_errors)
    sample_test_mse_raw = sample_test_mse * std_val**2

    print(f"\nTest MSE (normalized)     : {sample_test_mse:.6f} | "
          f"Ratio: {sample_test_mse/sigma**2:.4f}")
    print(f"Test MSE (original scale) : {sample_test_mse_raw:.4f}")

    # ── Forecast plot in original scale ──────────────────────────
    predictions_raw = np.array(predictions) * std_val + mean_val
    true_values_raw = np.array(true_values) * std_val + mean_val
    steps           = np.arange(1, len(predictions_raw) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(steps, true_values_raw, label='True',
                 color='blue',   alpha=0.7)
    axes[0].plot(steps, predictions_raw, label='Predicted',
                 color='orange', alpha=0.7, linestyle='--')
    axes[0].set_title('Mamba Forecast — Nile River Test Set vs Predicted')
    axes[0].set_xlabel('Forecast Step')
    axes[0].set_ylabel('Volume (original scale)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    sq_errors_raw = (predictions_raw - true_values_raw) ** 2
    axes[1].bar(steps, sq_errors_raw, color='red', alpha=0.5,
                label='Squared Error')
    axes[1].set_title('Squared Error per Forecast Step (original scale)')
    axes[1].set_xlabel('Forecast Step')
    axes[1].set_ylabel('Squared Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mamba_nile_forecast_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Forecast plot saved to mamba_nile_forecast_plot.png")

    del embed, mamba, head, optimizer
    torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n── Mamba Nile Summary ───────────────────────────────────────")
print(f"Train MSE (normalized)     : {all_sample_mse[0]:.6f} | "
      f"Ratio: {all_sample_mse[0]/sigma**2:.4f}")
print(f"Train MSE (original scale) : {all_sample_mse[0] * std_val**2:.4f}")
print(f"Test  MSE (normalized)     : {sample_test_mse:.6f} | "
      f"Ratio: {sample_test_mse/sigma**2:.4f}")
print(f"Test  MSE (original scale) : {sample_test_mse_raw:.4f}")
print(f"sigma^2 (normalized)       : {sigma**2:.6f}")

# ── Save results ──────────────────────────────────────────────────────
torch.save({
    'train_mse_norm' : all_sample_mse[0],
    'train_mse_raw'  : all_sample_mse[0] * std_val**2,
    'test_mse_norm'  : sample_test_mse,
    'test_mse_raw'   : sample_test_mse_raw,
    'sigma'          : sigma,
    'sigma_raw'      : sigma * std_val,
    'mean_val'       : mean_val,
    'std_val'        : std_val,
    'T_train'        : T_train,
    'T_test'         : T_test,
}, 'mamba_nile_results.pt')

print("Results saved to mamba_nile_results.pt")
