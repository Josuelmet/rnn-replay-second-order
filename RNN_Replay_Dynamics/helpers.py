import numpy as np
import os
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn

import datasets
from model import NoisyRNN, sample_plaus


def gather_seeds(*hyperparam_strings, model_dir='results'):
    '''
    Example: gather_seeds('tmaze_dataset', 'unmask_every_3')
    '''
    seeds = []
    for f in os.listdir(model_dir):
        if all([string in f for string in hyperparam_strings]) and '.pt' in f:
            seeds.append(int( f.split('seed_')[-1].split('__')[0] ))
    return seeds



def simulate_models(sigma_s, hidden_dim, T_multiplier, seeds,
                    model_dir, model_prefix, param_combos,
                    plaus=False):
    '''
    Example: simulate_models(
                 sigma_s=0.05, hidden_dim=20, T_multiplier=1,
                 seeds=seeds, param_combos=param_combos,
                 model_dir='results', model_prefix='tmaze_dataset__unmask_every_3')
    '''
    dataset = model_prefix.split('_')[0]
    assert dataset in ['tmaze', 'triangle'], f"dataset {dataset} is invalid"
    if dataset == 'tmaze':
        dataset_fn, groups = datasets.tmaze, 2
    elif dataset == 'triangle':
        dataset_fn, groups = datasets.triangle, 6
        
    replays = {str(d): [] for d in param_combos}
    awake_trajectories = []
    
    for seed in tqdm(seeds):
        # Load randomly noisy initial positions
        s = dataset_fn(seed=seed, sigma_s=sigma_s) 
        awake_trajectories.append(s)
        init_pos = torch.from_numpy(s[0].reshape(-1,2)).float()

        # Load model parameters
        file = os.path.join(model_dir, model_prefix) + f'__seed_{seed:02d}'
        my_rnn = NoisyRNN(d=2, hidden_dim=hidden_dim,
                          act=nn.Tanh() if "tanh" in model_prefix else nn.LeakyReLU(),
                          use_norm=False)
        my_rnn.load_state_dict(torch.load(file + '__model.pt'))
        intentions = torch.from_numpy(np.load(file + '__extra.npz')['intentions']).float()

        # Simulate the model...
        T = s.shape[0] * T_multiplier
        N = intentions.shape[0]
        # ...with a variety of parameters
        if plaus:
            print("Running sample_plaus() instead of rnn.sample()")
        for kwargs in param_combos:
            if plaus:
                x_hat, r, = sample_plaus(my_rnn,
                    T=T, N=N, init_pos=init_pos, intentions=intentions, **kwargs
                )
            else:
                x_hat, r = my_rnn.sample(
                    T=T, N=N, init_pos=init_pos, intentions=intentions, **kwargs
                )
            x_hat = x_hat.detach().reshape(T, groups, N//groups, 2)
            replays[str(kwargs)].append(x_hat)

    # each v shape = (T*T_multiplier, groups, N//groups, 2)
    # stacked v shape = (len(seeds), T*T_multiplier, groups, N//groups, 2)
    # replays shape = (T*T_multiplier, groups, len(seeds), N//groups, 2)
    replays = {k: torch.stack(v).transpose(0,1).transpose(1,2) for (k,v) in replays.items()}
    # shape = (T, groups, len(seeds)*N//groups, 2)
    awake_trajectories = np.concatenate(list(awake_trajectories), axis=2)
    return replays, torch.from_numpy(awake_trajectories).float()





def reach_stats(replay, endpoints, radius):
    # Tensor replay shape = (timesteps, groups, num samples, 2)
    # Tensor endpoints shape = (groups, 2)   
    groups = len(endpoints)
    assert replay.ndim == 4 and replay.shape[1] == groups
    reach_times = []
    failure_rate = 0
    for group in range(groups):
        distance = ((replay[:,group] - endpoints[group]) ** 2).sum(-1).sqrt()
        group_times = (distance <= radius).int().argmax(0)
        reach_times.append(group_times[group_times != 0])
        failure_rate += (group_times == 0).float().mean().item() / groups
    reach_times = torch.cat(reach_times).float()
    return reach_times, reach_times.median().item(), reach_times.mean().item(), \
           reach_times.std().item(), failure_rate


def dict_reach_stats(replay_dict, endpoints, radius):
    rts, rt_meds, rt_avgs, rt_stds, failure_rates = {}, {}, {}, {}, {}
    for k, replay in replay_dict.items(): # k = key: the string of kwarg param combos
        rts[k], rt_meds[k], rt_avgs[k], rt_stds[k], failure_rates[k] = \
            reach_stats(replay, endpoints, radius)
    return rts, rt_meds, rt_avgs, rt_stds, failure_rates


# for TikZ
def dict_to_csv(param_dict, col_param, path=None):
    df1 = pd.DataFrame([eval(k) for k in param_dict.keys()])
    df2 = df1.copy()
    df1['value'] = param_dict.values()

    df2.pop(col_param)
    df2.drop_duplicates(ignore_index=True, inplace=True)
    
    for val, mini_df in df1.groupby(col_param):
        df2[f'{col_param}={val}'] = list(mini_df['value'])
        
    if path is not None:
        df2.to_csv(path)
    return df2


# for TikZ
def array_to_heatmap(matrix, path=None):
    mat = matrix.T # transpose for TikZ
    x = np.arange(mat.shape[0]).repeat(mat.shape[1])
    y = np.tile(np.arange(mat.shape[1]), mat.shape[0])
    output = np.vstack((x, y, mat.flatten())).T # tall matrix of shape (# elements in mat) x 3
    if path is not None:
        pd.DataFrame(output).to_csv(path, sep=' ', index=False, header=False)
        print('Make sure to add newlines before plotting in TikZ')
    return output