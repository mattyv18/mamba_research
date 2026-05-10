import torch
import torch.nn.functional as F
import math
import numpy as np
from einops import repeat

def test_seed(seed, d_model=16, d_state=16, d_inner=32, d_conv=4):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dt_rank = math.ceil(d_model / 16)

    params = {}
    params['W_embed']     = torch.randn(d_model, 1) * 0.5
    params['b_embed']     = torch.zeros(d_model)
    params['W_ssm']       = torch.randn(d_inner, d_model) * (1.0 / math.sqrt(d_model))
    params['W_gate']      = torch.randn(d_inner, d_model) * (1.0 / math.sqrt(d_model))
    params['conv_weight'] = torch.randn(d_inner, d_conv) * (1.0 / math.sqrt(d_conv))
    params['conv_bias']   = torch.zeros(d_inner)
    params['b_gate']      = torch.ones(d_inner) * 2.0
    params['W_proj']      = torch.randn(dt_rank + 2*d_state, d_inner) * (3.0 / math.sqrt(d_inner))

    dt_init_std       = dt_rank ** -0.5
    params['W_delta'] = torch.zeros(d_inner, dt_rank).uniform_(-dt_init_std, dt_init_std)
    dt = torch.exp(
        torch.rand(d_inner) * (math.log(0.5) - math.log(0.001)) + math.log(0.001)
    ).clamp(min=1e-4)
    params['b_delta'] = dt + torch.log(-torch.expm1(-dt))

    A = repeat(torch.linspace(0.01, 0.5, d_state), "n -> d n", d=d_inner)
    params['A_log'] = torch.log(A)
    params['D']     = torch.ones(d_inner)
    params['W_out'] = torch.randn(d_model, d_inner) * (1.0 / math.sqrt(d_inner))
    params['W_head']= torch.randn(1, d_model) * (1.0 / math.sqrt(d_model))

    # quick test — run 1 sample for 200 steps
    h           = torch.zeros(d_inner, d_state)
    conv_buffer = torch.zeros(d_inner, d_conv)
    x_t         = torch.tensor(0.1)
    hats        = []

    for t in range(200):
        v_t    = params['W_embed'] @ x_t.unsqueeze(0) + params['b_embed']
        p_ssm  = params['W_ssm']  @ v_t
        p_gate = params['W_gate'] @ v_t + params['b_gate']

        conv_buffer = torch.roll(conv_buffer, -1, dims=1)
        conv_buffer[:, -1] = p_ssm
        u_ssm = (conv_buffer * params['conv_weight']).sum(dim=1) + params['conv_bias']
        u_ssm = F.silu(u_ssm)

        x_dbl     = params['W_proj'] @ u_ssm
        delta_raw = x_dbl[:dt_rank]
        B_t       = x_dbl[dt_rank: dt_rank + d_state]
        C_t       = x_dbl[dt_rank + d_state:]

        delta_t = F.softplus(params['W_delta'] @ delta_raw + params['b_delta'])
        A       = -torch.exp(params['A_log'])
        A_bar   = torch.exp(delta_t.unsqueeze(1) * A)
        B_bar   = delta_t.unsqueeze(1) * B_t.unsqueeze(0)
        h       = A_bar * h + B_bar * u_ssm.unsqueeze(1)
        y_t     = (C_t.unsqueeze(0) * h).sum(dim=1) + params['D'] * u_ssm
        y_t     = F.silu(p_gate) * y_t
        o_t     = params['W_out'] @ y_t
        x_next_hat = (params['W_head'] @ o_t).squeeze()

        if torch.isnan(x_next_hat):
            return None

        hats.append(x_next_hat.item())
        x_t = x_next_hat + 0.001 * torch.randn(1).item()

    hats = np.array(hats)
    mean = abs(hats.mean())
    std  = hats.std()
    lag1 = np.corrcoef(hats[:-1], hats[1:])[0,1]

    return {'seed': seed, 'mean': mean, 'std': std, 'lag1': lag1}

# search over seeds
print("Searching for good initialization...")
for seed in range(500):
    result = test_seed(seed)
    if result is None:
        continue
    if (result['mean'] < 0.01 and
        result['std'] > 0.005 and
        result['lag1'] > 0.2):
        print(f"Seed {seed:>4d} | mean: {result['mean']:.6f} | "
              f"std: {result['std']:.6f} | lag1: {result['lag1']:.4f}")
