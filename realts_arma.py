import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import torch
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from collections import Counter

# ── Reproducibility ───────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Load Nile data ────────────────────────────────────────────────────
nile   = sm.datasets.nile.load_pandas().data
series = nile['volume'].values.astype(np.float32)

print(f"Series length: {len(series)}")
print(f"Mean:          {series.mean():.4f}")
print(f"Std:           {series.std():.4f}")
print(f"Min:           {series.min():.4f}")
print(f"Max:           {series.max():.4f}")

# ── ADF test on unnormalized series ───────────────────────────────────
result = adfuller(series)
print(f"\nADF test (original scale):")
print(f"ADF statistic: {result[0]:.4f}")
print(f"p-value:       {result[1]:.4f}")
print(f"{'Stationary' if result[1] < 0.05 else 'Non-stationary'}")

# ── Plot unnormalized full series ─────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(series, color='blue', alpha=0.7)
plt.title('Nile River Annual Flow Volume (Original Scale)')
plt.xlabel('Year')
plt.ylabel('Volume')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('arma_nile_full_series_raw.png', dpi=150, bbox_inches='tight')
plt.close()
print("Unnormalized series plot saved to arma_nile_full_series_raw.png")

# ── ACF and PACF on unnormalized series ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series,  lags=40, ax=axes[0],
         title='Nile River ACF (Original Scale)')
plot_pacf(series, lags=40, ax=axes[1],
         title='Nile River PACF (Original Scale)')
plt.tight_layout()
plt.savefig('arma_nile_acf_pacf_raw.png', dpi=150, bbox_inches='tight')
plt.close()
print("Unnormalized ACF/PACF saved to arma_nile_acf_pacf_raw.png")

# ── Normalize ─────────────────────────────────────────────────────────
mean_val = series.mean()
std_val  = series.std()
series   = (series - mean_val) / std_val

print(f"\nAfter normalization:")
print(f"Mean: {series.mean():.4f}")
print(f"Std:  {series.std():.4f}")

# ── ADF test on normalized series ─────────────────────────────────────
result_norm = adfuller(series)
print(f"\nADF test (normalized):")
print(f"ADF statistic: {result_norm[0]:.4f}")
print(f"p-value:       {result_norm[1]:.4f}")
print(f"{'Stationary' if result_norm[1] < 0.05 else 'Non-stationary'}")

# ── Plot normalized full series ───────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(series, color='blue', alpha=0.7)
plt.title('Nile River Annual Flow Volume (Normalized)')
plt.xlabel('Year')
plt.ylabel('Volume (normalized)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('arma_nile_full_series_norm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Normalized series plot saved to arma_nile_full_series_norm.png")

# ── ACF and PACF on normalized series ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series,  lags=40, ax=axes[0],
         title='Nile River ACF (Normalized)')
plot_pacf(series, lags=40, ax=axes[1],
         title='Nile River PACF (Normalized)')
plt.tight_layout()
plt.savefig('arma_nile_acf_pacf_norm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Normalized ACF/PACF saved to arma_nile_acf_pacf_norm.png")

# ── Train test split ──────────────────────────────────────────────────
T_total = len(series)
T_train = int(T_total * 0.9)
T_test  = T_total - T_train

train_data = series[:T_train]
test_data  = series[T_train:]

print(f"\nT_total : {T_total}")
print(f"T_train : {T_train}")
print(f"T_test  : {T_test}")

# ── Sigma estimate ────────────────────────────────────────────────────
diffs = np.diff(series)
sigma = diffs.std() / math.sqrt(2)
print(f"sigma   : {sigma:.6f}")
print(f"sigma^2 : {sigma**2:.6f}")

# ── Fit ARMA on training data ─────────────────────────────────────────
print("\nFitting ARMA to Nile training data...")

model = pm.auto_arima(
    train_data,
    d                     = 0,
    start_p               = 0,
    max_p                 = 15,
    start_q               = 0,
    max_q                 = 5,
    information_criterion = 'aic',
    stepwise              = True,
    suppress_warnings     = True,
    error_action          = 'ignore'
)

print(f"Selected order: {model.order}")

# ── Training MSE ──────────────────────────────────────────────────────
fitted       = model.fittedvalues()
train_errors = (fitted - train_data) ** 2
train_mse    = train_errors.mean()
train_mse_raw = train_mse * std_val**2

print(f"\nTrain MSE (normalized)     : {train_mse:.6f} | Ratio: {train_mse/sigma**2:.4f}")
print(f"Train MSE (original scale) : {train_mse_raw:.4f}")

# ── Training fit plot ─────────────────────────────────────────────────
fitted_raw     = fitted * std_val + mean_val
train_data_raw = train_data * std_val + mean_val

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(train_data_raw, label='True', color='blue', alpha=0.7)
axes[0].plot(fitted_raw,     label='Fitted', color='orange', alpha=0.7)
axes[0].set_title('ARMA Fit — Nile River Full Training Sequence')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Volume')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_data_raw[-50:], label='True', color='blue', alpha=0.7)
axes[1].plot(fitted_raw[-50:],     label='Fitted', color='orange', alpha=0.7)
axes[1].set_title('ARMA Fit — Last 50 Timesteps')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Volume')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('arma_nile_fit.png', dpi=150, bbox_inches='tight')
plt.close()
print("Training fit plot saved to arma_nile_fit.png")

# ── Rolling forecast on test set ──────────────────────────────────────
print("\nRunning ARMA rolling forecast on test set...")

predictions    = []
true_values    = []
test_errors    = []
best_order     = model.order

for t in range(T_test - 1):

    current_train = series[:T_train + t]

    try:
        if t == 0:
            refit = pm.auto_arima(
                current_train,
                d                     = 0,
                start_p               = 0,
                max_p                 = 15,
                start_q               = 0,
                max_q                 = 5,
                information_criterion = 'aic',
                stepwise              = True,
                suppress_warnings     = True,
                error_action          = 'ignore'
            )
            best_order = refit.order
        else:
            refit = pm.auto_arima(
                current_train,
                d                     = 0,
                start_p               = best_order[0],
                max_p                 = best_order[0],
                start_q               = best_order[2],
                max_q                 = best_order[2],
                information_criterion = 'aic',
                stepwise              = True,
                suppress_warnings     = True,
                error_action          = 'ignore'
            )

        x_pred = refit.predict(n_periods=1)[0]
        x_true = series[T_train + t]
        error  = (x_pred - x_true) ** 2

        predictions.append(x_pred)
        true_values.append(x_true)
        test_errors.append(error)

        pred_raw = x_pred * std_val + mean_val
        true_raw = x_true * std_val + mean_val

        print(f"Step {t+1:>3d}/{T_test-1} | "
              f"Order: {best_order} | "
              f"Pred: {pred_raw:.2f} | "
              f"True: {true_raw:.2f} | "
              f"SE (norm): {error:.6f}")

    except Exception as e:
        print(f"Step {t+1} failed: {e}")
        test_errors.append(float('nan'))

# ── Test MSE ──────────────────────────────────────────────────────────
test_errors    = np.array(test_errors)
test_errors    = test_errors[~np.isnan(test_errors)]
test_mse       = test_errors.mean()
test_mse_raw   = test_mse * std_val**2

print(f"\nTest MSE (normalized)     : {test_mse:.6f} | Ratio: {test_mse/sigma**2:.4f}")
print(f"Test MSE (original scale) : {test_mse_raw:.4f}")

# ── Forecast plot ─────────────────────────────────────────────────────
predictions_raw = np.array(predictions) * std_val + mean_val
true_values_raw = np.array(true_values) * std_val + mean_val
steps           = np.arange(1, len(predictions_raw) + 1)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(steps, true_values_raw,  label='True',
             color='blue',   alpha=0.7)
axes[0].plot(steps, predictions_raw,  label='Predicted',
             color='orange', alpha=0.7, linestyle='--')
axes[0].set_title('ARMA Forecast — Nile River Test Set vs Predicted')
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
plt.savefig('arma_nile_forecast_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Forecast plot saved to arma_nile_forecast_plot.png")

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n── ARMA Nile Summary ────────────────────────────────────────")
print(f"Selected order             : {model.order}")
print(f"Train MSE (normalized)     : {train_mse:.6f} | Ratio: {train_mse/sigma**2:.4f}")
print(f"Train MSE (original scale) : {train_mse_raw:.4f}")
print(f"Test  MSE (normalized)     : {test_mse:.6f} | Ratio: {test_mse/sigma**2:.4f}")
print(f"Test  MSE (original scale) : {test_mse_raw:.4f}")
print(f"sigma^2 (normalized)       : {sigma**2:.6f}")

# ── Save results ──────────────────────────────────────────────────────
np.save('arma_nile_results.npy', {
    'train_mse_norm' : train_mse,
    'train_mse_raw'  : train_mse_raw,
    'test_mse_norm'  : test_mse,
    'test_mse_raw'   : test_mse_raw,
    'sigma'          : sigma,
    'sigma_raw'      : sigma * std_val,
    'mean_val'       : mean_val,
    'std_val'        : std_val,
    'order'          : model.order,
    'T_train'        : T_train,
    'T_test'         : T_test,
})

print("Results saved to arma_nile_results.npy")
