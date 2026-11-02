import argparse
import numpy as np
import os
from tqdm import tqdm

import torch
from torch import nn

import datasets
from model import NoisyRNN

# 1.
# -----------------------
# Command line parameters
# -----------------------
parser = argparse.ArgumentParser("train")
parser.add_argument("--seed", help="Random seed for dataset and model initialization", type=int)
parser.add_argument("--dataset", help="Dataset choice: 'triangle', 't-maze', or 'ou'", type=str)
parser.add_argument("--unmask_every", help="Maximum difficulty of curriculum learning. 1 = use all observations, 2 = every other observation, 3 = every 3rd observation, etc.", type=int)
parser.add_argument("--noleak", help="Use leakage: True/False.", action='store_true', default=False)
parser.add_argument("--activation", help="Activation function: 'relu' or 'tanh'", default="relu", type=str)
args = parser.parse_args()


args.dataset = args.dataset.lower()
assert args.dataset in ['triangle', 'tmaze', 'ou'], f"dataset {dataset} is invalid"


# 2.
# ----------------------------------------------------------
# Initialize dataset and dataset-based hard-coded parameters
# ----------------------------------------------------------

if args.dataset == 'triangle':
    # Hard-coded parameters
    hidden_dim = 40
    sigma_s = 1
    sigma_r = sigma_s
    sigma_intention = sigma_r
    unmasked_epochs, masked_epochs = 20_000, 5_000
    # Initialize dataset
    s = datasets.triangle(seed=args.seed, sigma_s=sigma_s)   
elif args.dataset == 'tmaze':
    hidden_dim = 20
    sigma_s = 0.05
    sigma_r = sigma_s
    sigma_intention = sigma_r * 4
    unmasked_epochs, masked_epochs = 12_000, 3_000
    s = datasets.tmaze(seed=args.seed, sigma_s=sigma_s)
else:
    print(f"Dataset {args.dataset} has not been implemented yet.")
          
          
# 3.
# -----------------------
# Set up model and inputs
# -----------------------
torch.manual_seed(args.seed)    
#my_rnn = NoisyRNN(d=2, hidden_dim=hidden_dim, act=nn.LeakyReLU(),
my_rnn = NoisyRNN(d=2, hidden_dim=hidden_dim,
                  act=nn.Tanh() if args.activation == 'tanh' else nn.LeakyReLU(),
                  use_norm=False, use_leak=not args.noleak)
          
# initialize the intention vector
T, groups, N_per_group, _ = s.shape # shape = (T, groups, N//groups, d=2)
intentions = torch.randn(groups, hidden_dim).repeat_interleave(N_per_group, 0) * sigma_intention

# Initialize input observations
s_tensor = torch.from_numpy(s).float().reshape((s.shape[0], -1, s.shape[-1]))
observations = torch.zeros_like(s_tensor)
observations[1:] = s_tensor.diff(dim=0)
    
# 4.
# -----------------------------
# Train via curriculum learning
# -----------------------------
losses = []
for unmask_every in torch.arange(args.unmask_every).int() + 1:
    optim = torch.optim.Adam(my_rnn.parameters(), lr=1e-4)
    
    pbar = tqdm(range(unmasked_epochs if unmask_every == 1 else masked_epochs))
    for e in pbar:
        x_hat, r = my_rnn.sample(
            T=T, N=N_per_group*groups, sigma_r=sigma_r,
            init_pos=s_tensor[0], intentions=intentions, 
            observations=observations*(torch.arange(T).reshape(-1,1,1) % unmask_every == 0)
        )
        loss = (s_tensor - x_hat).abs().mean()

        pbar.set_description(f"Loss: {loss.item() :.5f}")
        losses.append(loss.item())
        loss.backward()
        optim.step()
        optim.zero_grad()
    
    
# 5.
# ----------
# Save model
# ----------
os.makedirs('results', exist_ok=True)
fname = f"{args.dataset}_dataset__unmask_every_{args.unmask_every}"
fname += "__noleak" if args.noleak else ""
fname += "__tanh" if args.activation == "tanh" else ""
fname += f"__seed_{args.seed:02d}"
torch.save(my_rnn.state_dict(), f"results/{fname}__model.pt")
# Save loss
np.savez(f"results/{fname}__extra",
        loss=np.asarray(losses),
        intentions=intentions.numpy()
)