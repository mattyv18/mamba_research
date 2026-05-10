# ── Imports ───────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import torch
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for HPC cluster
import matplotlib.pyplot as plt
from einops import repeat
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


# ── Reproducibility ───────────────────────────────────────────────────
# seed 333 was selected via seed search to produce stationary series
# with SNR ~2.3 and lag-1 correlation ~0.70
torch.manual_seed(333)
np.random.seed(333)
random.seed(333)
torch.cuda.manual_seed_all(333)

# ── Parameter initialization ──────────────────────────────────────────
# initializes the fixed DGP parameters theta
# these are the true parameters of the data generating process
# they remain fixed throughout — only the noise epsilon changes across samples
def initialize_params(d_model, d_state, d_inner, d_conv, dt_rank,
                      dt_min=0.001, dt_max=0.5, dt_init_floor=1e-4):
    params = {}

    # Step 1 — embedding weights
    # W_embed: projects scalar x_t to d_model dimensional vector
    # b_embed: zero bias prevents systematic drift in generated series
    params['W_embed'] = torch.randn(d_model, 1) * 0.5
    params['b_embed'] = torch.zeros(d_model)

    # Step 2 — expand and split weights
    # W_ssm:  projects embedding to SSM branch (d_inner,)
    # W_gate: projects embedding to gate branch (d_inner,)
    params['W_ssm']  = torch.randn(d_inner, d_model) * (1.0 / math.sqrt(d_model))
    params['W_gate'] = torch.randn(d_inner, d_model) * (1.0 / math.sqrt(d_model))

    # Step 3 — depthwise conv weights
    # conv_weight: (d_inner, d_conv) — one filter per channel
    # b_gate: large positive bias keeps gate open, allowing signal to flow
    params['conv_weight'] = torch.randn(d_inner, d_conv) * (1.0 / math.sqrt(d_conv))
    params['conv_bias']   = torch.zeros(d_inner)
    params['b_gate'] = torch.ones(d_inner) * 2.0

    # Step 4 — joint projection
    # W_proj: produces delta_raw, B_t, C_t from SSM input
    # scale 6.0 chosen to produce sufficient signal strength
    params['W_proj'] = torch.randn(dt_rank + 2 * d_state, d_inner) * (6.0 / math.sqrt(d_inner))

    # Step 5 — delta projection
    # W_delta: low rank bottleneck projection (d_inner, dt_rank)
    # b_delta: initialized to produce delta values in [dt_min, dt_max]
    dt_init_std = dt_rank ** -0.5
    params['W_delta'] = torch.zeros(d_inner, dt_rank).uniform_(-dt_init_std, dt_init_std)
    dt = torch.exp(
        torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    params['b_delta'] = dt + torch.log(-torch.expm1(-dt))

    # Step 5 — delta projection
    # W_delta: low rank bottleneck projection (d_inner, dt_rank)
    # b_delta: initialized to produce delta values in [dt_min, dt_max]
    A = repeat(
    torch.linspace(0.01, 0.5, d_state),   # values 0.01 to 0.5 instead of 1 to 16
    "n -> d n", d=d_inner
    )
    params['A_log'] = torch.log(A)

    # D skip connection — passes input directly to output
    params['D'] = torch.ones(d_inner)

    # Steps 10-11 — output projection and prediction head
    # W_out: scale 1.2 chosen to give SNR ~2.3 without causing explosion
    # W_head: centered to prevent systematic drift in predictions
    params['W_out']  =torch.randn(d_model, d_inner) * (1.2 / math.sqrt(d_inner))
    params['W_head'] = torch.randn(1, d_model) * (1.0 / math.sqrt(d_model))
    params['W_head'] = params['W_head'] - params['W_head'].mean()
    return params

# ── Sample generation ─────────────────────────────────────────────────
# generates n_samples time series of length T using the fixed DGP params
# each series is an independent realization of the same DGP
# T_burnin: discard first 50 steps to remove dependence on x0
def generate_samples(n_samples, T, sigma, params,
                     d_model, d_state, d_inner, d_conv, dt_rank,
                     T_burnin=50, x0=0.1):

    T_total = T + T_burnin
    all_samples = torch.zeros(n_samples, T)

    for s in range(n_samples):

        # initialize hidden state and conv buffer to zero at start of each series
        h           = torch.zeros(d_inner, d_state)
        conv_buffer = torch.zeros(d_inner, d_conv)
        x_t         = torch.tensor(x0)
        x_series    = []
        hats        = []
        for t in range(T_total):

            # Step 1 — embed scalar input to d_model vector
            v_t = params['W_embed'] @ x_t.unsqueeze(0) + params['b_embed']   # (d_model,)

            # Step 2 — expand to SSM and gate branches
            p_ssm  = params['W_ssm']  @ v_t                     # (d_inner,)
            p_gate = params['W_gate'] @ v_t + params['b_gate']                     # (d_inner,)

            # Step 3 — causal depthwise conv on SSM branch
            # conv_buffer maintains last d_conv values for each channel
            conv_buffer = torch.roll(conv_buffer, -1, dims=1)
            conv_buffer[:, -1] = p_ssm
            u_ssm = (conv_buffer * params['conv_weight']).sum(dim=1) \
                    + params['conv_bias']                        # (d_inner,)
            u_ssm = F.silu(u_ssm)   # SiLU nonlinearity                            # (d_inner,)


            # Step 4 — joint projection produces delta_raw, B_t, C_t
            x_dbl    = params['W_proj'] @ u_ssm                 # (dt_rank + 2*d_state,)
            delta_raw = x_dbl[:dt_rank]                         # (dt_rank,)
            B_t       = x_dbl[dt_rank: dt_rank + d_state]       # (d_state,)
            C_t       = x_dbl[dt_rank + d_state:]               # (d_state,)


            # Step 5 — compute input dependent delta via softplus
            # delta controls memory vs forgetting tradeoff
            delta_t = F.softplus(
                params['W_delta'] @ delta_raw + params['b_delta']
            )                                                    # (d_inner,)


            # Step 6 — exponential Euler discretization
            # A_bar: discrete state transition — between 0 and 1
            # B_bar: discrete input projection
            A     = -torch.exp(params['A_log'])                  # (d_inner, d_state)
            A_bar = torch.exp(delta_t.unsqueeze(1) * A)          # (d_inner, d_state)
            B_bar = delta_t.unsqueeze(1) * B_t.unsqueeze(0)      # (d_inner, d_state)



            # Step 7 — hidden state update
            # h_t = A_bar * h_{t-1} + B_bar * u_ssm
            h = A_bar * h + B_bar * u_ssm.unsqueeze(1)          # (d_inner, d_state)


            # Step 8 — readout with D skip connection
            y_t = (C_t.unsqueeze(0) * h).sum(dim=1)             # (d_inner,)
            y_t = y_t + params['D'] * u_ssm                     # (d_inner,)


            # Step 9 — gating with SiLU
            y_t = F.silu(p_gate) * y_t                          # (d_inner,)

            # Step 10 — output projection  to d_model
            o_t = params['W_out'] @ y_t                         # (d_model,)

            # Step 11 — prediction head to scalar
            x_next_hat = (params['W_head'] @ o_t).squeeze()
            hats.append(x_next_hat.item())


            # metrics used to help tweak weights
            if t in [0, 10, 25, 50, 55, 59]:
                print(f"t={t:>3d} | h std: {h.std().item():.6f} | "
                      f"x_next_hat: {x_next_hat.item():.6f}")

            # draw next observation — predicted value plus Gaussian noise
            # sigma controls SNR — set to signal_std / 2 for SNR = 2
            x_next =  x_next_hat + sigma  * torch.randn(1).item()

            # only keep post-burnin values
            if t >= T_burnin:
                x_series.append(x_next.item())

            x_t = x_next
        all_samples[s] = torch.tensor(x_series)
        hats = torch.tensor(hats)
        print(f"x_next_hat mean: {hats.mean():.6f}")
        print(f"x_next_hat std:  {hats.std():.6f}")
        print(f"x_next_hat abs mean: {hats.abs().mean():.6f}")
    return all_samples

# ── Hyperparameters ───────────────────────────────────────────────────
# architecture dimensions must match what training scripts expect
d_model   = 16 # embedding dimension
d_state   = 16 # SSM state dimension N
d_inner   = 32 # expanded inner dimension E * d_model, E=2
d_conv    = 4 # depthwise conv kernel size
dt_rank   = math.ceil(d_model / 16) # delta projection rank r
T         = 1200 # total series length
n_samples = 500 # number of independent series to generate
T_burnin  = 50 # burnin steps discarded to remove x0 dependence
x0        = 0.1 # initial value for all series
n_epochs  = 3000 #epochs
lr        = 1e-3 #learning rate
T_train    = 1150 # training sequence length
T_test     = 50 # test sequence length

# ── Initialize DGP parameters ─────────────────────────────────────────
# theta is fixed — same parameters used for all 500 samples
# only the noise realizations differ across samples
params  = initialize_params(d_model, d_state, d_inner, d_conv, dt_rank)

# ── Pilot run ─────────────────────────────────────────────────────────
# run with tiny noise to measure the pure signal variance
# used to calibrate sigma so that SNR = signal_std / sigma = 2.0
pilot_sigma = 0.001
print("Running pilot to measure signal scale...")
pilot = generate_samples(
    n_samples = 10,
    T         = 500,
    sigma     = pilot_sigma,
    params    = params,
    d_model   = d_model,
    d_state   = d_state,
    d_inner   = d_inner,
    d_conv    = d_conv,
    dt_rank   = dt_rank,
    T_burnin  = T_burnin,
    x0        = x0
)


# estimate signal variance by subtracting known noise variance
total_var  = pilot.var(dim=1).mean().item()
noise_var  = pilot_sigma ** 2
signal_var = max(total_var - noise_var, 0.0)
signal_std = math.sqrt(signal_var)

print(f"Total std  : {math.sqrt(total_var):.6f}")
print(f"Noise std  : {pilot_sigma:.6f}")
print(f"Signal std : {signal_std:.6f}")


# ── Set sigma for SNR = 2.0 ───────────────────────────────────────────
# sigma = signal_std / 2 gives signal variance = 4 * sigma^2
# meaning signal is 4x stronger than noise variance
if signal_std < 1e-6:
    print("WARNING — signal still zero, check initialization")
    sigma = pilot_sigma / 2.0
else:
    sigma = signal_std / 2.0
    print(f"Setting sigma to: {sigma:.6f}")
    print(f"Target SNR: 2.0")

# ── Generate 500 samples ──────────────────────────────────────────────
print("Generating samples...")
samples = generate_samples(n_samples, T, sigma, params,
                           d_model, d_state, d_inner, d_conv, dt_rank,
                           T_burnin=T_burnin, x0=0.1)

# metrics to help choose weights
print(samples.shape)  # (500, 201)
actual_snr = samples.std(dim=1).mean().item() / sigma
print(f"Actual SNR: {actual_snr:.4f}")   # should be close to 5


# ── Train test split ──────────────────────────────────────────────────
train_data = samples[:, :T_train]
test_data  = samples[:, T_train:]

# ── Diagnostics ───────────────────────────────────────────────────────
print(f"sigma        = {sigma:.10f}")
print(f"sigma^2      = {sigma**2:.10f}")
print(f"signal std   = {samples.std(dim=1).mean().item():.6f}")
print(f"SNR          = {samples.std(dim=1).mean().item() / sigma:.4f}")
signal_std = train_data.std(dim=1).mean().item()
print(f"Signal std : {signal_std:.6f}")
print(f"Noise sigma: {sigma:.6f}")
print(f"SNR        : {signal_std / sigma:.4f}")
import numpy as np
seq = train_data[0].numpy()
print("Lag-1 corr:", np.corrcoef(seq[:-1], seq[1:])[0,1])
plt.figure()

for i in range(1):  # plot first 5 samples
    plt.plot(samples[i].numpy(), alpha=0.7)

# ── Plot one generated series ─────────────────────────────────────────
plt.title("Multiple Generated Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig("timeseries_final2.png", dpi=300, bbox_inches="tight")
plt.close()

# ── ACF and PACF plots for 3 samples ─────────────────────────────────
# ACF shows autocorrelation structure — should decay slowly given lag-1 ~0.70
# PACF shows partial autocorrelation — cutoff indicates AR order
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

for i in range(3):
    plot_acf(train_data[i].numpy(),  lags=40, ax=axes[i, 0],
             title=f'Sample {i+1} ACF')
    plot_pacf(train_data[i].numpy(), lags=40, ax=axes[i, 1],
             title=f'Sample {i+1} PACF')

plt.tight_layout()
plt.savefig('acf_pacf_final2.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Mean of series means: {train_data.mean(dim=1).mean():.4f}")
print(f"Mean of series stds:  {train_data.std(dim=1).mean():.4f}")
print(f"Noise sigma:          {sigma:.4f}")

# ── Save DGP data ─────────────────────────────────────────────────────
# save all data and hyperparameters so training scripts can load them
# all training and test scripts load from this single file
torch.save({
    'samples'    : samples,
    'train_data' : train_data,
    'test_data'  : test_data,
    'sigma'      : sigma,
    'T_train'    : T_train,
    'T_test'     : T_test,
    'n_samples'  : n_samples,
    'd_model'    : d_model,
    'd_state'    : d_state,
    'd_inner'    : d_inner,
    'd_conv'     : d_conv,
    'dt_rank'    : dt_rank,
}, 'dgp_data_v2.pt')

print("DGP data saved to dgp_data.pt")


# ── ADF stationarity tests on 5 samples ──────────────────────────────
# confirms generated series are stationary — p-value < 0.05 means stationary
# important since ARMA assumes stationarity
print("\n── ADF Stationarity Tests ───────────────────────────────────")
for i in range(5):
    seq    = train_data[i].numpy()
    result = adfuller(seq)
    print(f"Sample {i+1} | ADF: {result[0]:.4f} | p-value: {result[1]:.4f} | "
          f"{'Stationary' if result[1] < 0.05 else 'Non-stationary'}")
