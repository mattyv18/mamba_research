# ── Imports ───────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import csv
import torch
import pmdarima as pm
from collections import Counter

# ── Reproducibility ───────────────────────────────────────────────────
# fix all random seeds so results are identical every run
# ARMA is deterministic so seeds mainly affect numpy internal operations
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Reproducibility ───────────────────────────────────────────────────
# fix all random seeds so results are identical every run
# ARMA is deterministic so seeds mainly affect numpy internal operations
print("Running ARMA rolling forecast on test set...")
data       = torch.load('dgp_data_v2.pt')
samples    = data['samples'] # (500, 1200) full sequences including test
sigma      = data['sigma'] # noise std — theoretical minimum is sigma^2
n_samples  = data['n_samples'] # 500
T_train    = data['T_train'] # training sequence length — 1150
T_test     = data['T_test'] # test sequence length — 50


# ── Pre-select ARMA order for each sample at step 0 ──────────────────
# order selection is expensive so we do it once per sample
# at subsequent steps we refit with the same fixed order
# this avoids running auto_arima 500 * 49 = 24500 times
# instead we run it 500 times for order selection + 500*48 cheap refits
print("Selecting ARMA orders for all samples at step 0...")
best_orders = {} # maps sample index to selected (p, d, q) order

for s in range(n_samples):
    sample        = samples[s].numpy()
    current_train = sample[:T_train] # use only training data for order selection

    try:
        # auto_arima searches over AR and MA orders
        # selects the order that minimizes AIC — balances fit vs complexity
        # d=0 — no differencing since DGP is stationary by design
        # max_p=20 — allow up to AR(20) to give ARMA its best chance
        # max_q=5  — allow up to MA(5)
        # stepwise=True — faster than exhaustive search
        model = pm.auto_arima(
            current_train,
            d                     = 0,
            start_p               = 0,
            max_p                 = 20,
            start_q               = 0,
            max_q                 = 5,
            information_criterion = 'aic',
            stepwise              = True,
            suppress_warnings     = True,
            error_action          = 'ignore'
        )
        best_orders[s] = model.order
    except Exception as e:
        print(f"Sample {s+1} order selection failed: {e}")
        best_orders[s] = (1, 0, 0)

    print(f"Sample {s+1:>4d}/{n_samples} | Order: {best_orders[s]}")

# ── Create CSV and write header ───────────────────────────────────────
# CSV saves per step metrics immediately after each step completes
# allows monitoring progress live while job is running
with open('arma_test_per_step2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'mean_mse', 'std_mse', 'mean_ratio', 'std_ratio'])

# storage for per step summary metrics
all_step_means = []
all_step_stds  = []

# ── Outer loop — forecast steps ───────────────────────────────────────
# step is the outer loop so we get per step metrics across all 500 samples
# at each step we compute mean and std MSE across all 500 samples
# this gives us ARMA performance as a function of forecast horizon
for step in range(T_test - 1):
    print(f"\n── Step {step+1}/{T_test-1} ──────────────────────────────────")

    # collect one squared error per sample for this step
    step_errors = []

    # ── Inner loop — samples ──────────────────────────────────────────
    for s in range(n_samples):

        # expanding window — use all data up to current forecast origin
        # at step 0: train on x_1...x_1150, predict x_1151
        # at step 1: train on x_1...x_1151, predict x_1152 etc.
        sample        = samples[s].numpy()
        current_train = sample[:T_train + step]
        order         = best_orders[s] # use pre-selected order for this sample

        try:
            # refit ARMA with fixed order on expanding window
            # fixing the order avoids expensive order selection at every step
            # start_p = max_p = order[0] forces exactly AR(p) order
            # start_q = max_q = order[2] forces exactly MA(q) order
            model = pm.auto_arima(
                current_train,
                d                     = 0,
                start_p               = order[0],
                max_p                 = order[0],
                start_q               = order[2],
                max_q                 = order[2],
                information_criterion = 'aic',
                stepwise              = True,
                suppress_warnings     = True,
                error_action          = 'ignore'
            )
            # predict one step ahead using fitted coefficients
            # and last p true values from expanding window
            x_pred = model.predict(n_periods=1)[0]
            x_true = sample[T_train + step]
            error  = (x_pred - x_true) ** 2
            step_errors.append(error)

            print(f"Step {step+1:>3d}/{T_test-1} | "
                  f"Sample {s+1:>4d}/{n_samples} | "
                  f"Order: {order} | "
                  f"Pred: {x_pred:.4f} | "
                  f"True: {x_true:.4f} | "
                  f"SE: {error:.6f}")

        except Exception as e:
            # record nan if fitting fails — removed before computing metrics
            print(f"Step {step+1} Sample {s+1} failed: {e}")
            step_errors.append(float('nan'))

# ── Per step metrics across all 500 samples ───────────────────────
    # after all 500 samples complete for this step
    # remove any nan values from failed fits
    # compute mean and std of squared errors across samples
    step_errors = np.array(step_errors)
    step_errors = step_errors[~np.isnan(step_errors)]
    step_mean   = step_errors.mean()
    step_std    = step_errors.std()

    all_step_means.append(step_mean)
    all_step_stds.append(step_std)

    print(f"\nStep {step+1} | "
          f"Mean MSE: {step_mean:.6f} | "
          f"Std: {step_std:.6f} | "
          f"Ratio: {step_mean/sigma**2:.4f}")


    # write this step's results to CSV immediately
    # appending one row after each step so progress is visible live
    with open('arma_test_per_step2.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            step + 1,
            step_mean,
            step_std,
            step_mean / sigma**2,
            step_std  / sigma**2
        ])


# ── Overall summary across all steps and samples ──────────────────────
# average the per step means to get overall rolling forecast MSE
# this is the primary metric for comparing Mamba vs ARMA on test data
all_step_means    = np.array(all_step_means)
all_step_stds     = np.array(all_step_stds)
overall_mean_afor = all_step_means.mean()
overall_std_afor  = all_step_means.std()

# count most commonly selected orders across all 500 samples
# consistent order selection indicates stable linear structure in DGP
order_counts = Counter(best_orders.values())
most_common  = order_counts.most_common(5)

# save results
np.save('arma_test_results2.npy', {
    'per_step_means'     : all_step_means,
    'per_step_stds'      : all_step_stds,
    'aic_rolling_orders' : list(best_orders.values()),
    'mean_aic_rolling'   : overall_mean_afor,
    'std_aic_rolling'    : overall_std_afor,
})

print(f"\n── ARMA Rolling Forecast Summary ────────────────────────────")
print(f"Mean MSE : {overall_mean_afor:.6f}")
print(f"Std  MSE : {overall_std_afor:.6f}")
print(f"Theoretical minimum (sigma^2) : {sigma**2:.6f}")
print(f"Ratio MSE / sigma^2           : {overall_mean_afor / sigma**2:.4f}")
print(f"\nMost commonly selected orders:")
for order, count in most_common:
    print(f"  {order} — selected {count} times")
print("Saved to arma_test_results2.npy and arma_test_per_step2.csv")
