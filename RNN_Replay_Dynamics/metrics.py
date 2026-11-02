import numpy as np
from scipy.linalg import sqrtm

import torch


def cov(trials):
    # activity should have shape T, num samples, 2
    assert trials.ndim == 3 and trials.shape[-1] == 2
    trials = trials.transpose(0,1).flatten(1) # T,N,2 -> N,T,2 -> N,T*2
    xxT = lambda x: x.outer(x)
    autocorrelation_of_mean = xxT(trials.mean(0))
    mean_of_autocorrelation = torch.stack([xxT(trial) for trial in trials]).mean(0)
    return mean_of_autocorrelation - autocorrelation_of_mean

def wasserstein_distance(m1, m2, C1, C2):
    sqrt_C2 = sqrtm(C2)
    return (m1 - m2).square().sum().item() + \
           np.trace(C1 + C2 - 2 * sqrtm(sqrt_C2 @ C1.numpy() @ sqrt_C2))

def kl_divergence(m1, m2, C1, C2):
    inv_C2 = torch.inverse(C2)
    return 0.5 * (
        (torch.det(C2) / torch.det(C1)).log() + \
        torch.trace(inv_C2 @ C1) + \
        (m2 - m1) @ inv_C2 @ (m2 - m1) - len(m1)
    )

def metric_trials(trials1, trials2, metric):
    assert metric in ['wd', 'kl']
    # shapes should be T, num samples, 2 for each argument
    assert trials1.ndim == trials2.ndim == 3 and trials1.shape[0] == trials2.shape[0] \
           and trials1.shape[-1] == trials2.shape[-1] == 2
    m1, m2 = trials1.mean(1).flatten(), trials2.mean(1).flatten()
    C1, C2 = cov(trials1), cov(trials2)
    return np.abs(kl_divergence(m1, m2, C1, C2) if metric == 'kl' else \
                  wasserstein_distance(m1, m2, C1, C2))

def calc_all_metric(replay_dict, awake, metric):
    assert awake.ndim == 4
    T_awake, groups = awake.shape[:2]
    #vals = {k: np.array([metric_trials(v[:T_awake, g], awake[:,g], metric) for g in range(groups)])
    #        for (k,v) in replay_dict.items()}
    #means = {k: np.mean(v) for (k,v) in replay_dict.items()}
    #return vals, means
    return {k: np.mean([
        metric_trials(v[:T_awake, g], awake[:,g], metric) for g in range(groups)
    ]) for (k,v) in replay_dict.items()}

'''
def wd_trials(trials1, trials2):
    # shapes should be T, num samples, 2 for each argument
    assert trials1.ndim == trials2.ndim == 3 and trials1.shape[0] == trials2.shape[0] \
           and trials1.shape[-1] == trials2.shape[-1] == 2
    return np.abs(wasserstein_distance(trials1.mean(1), trials2.mean(1),
                                       cov(trials1), cov(trials2)))

def calc_all_wds(replay_dict, awake):
    assert awake.ndim == 4
    T_awake, groups = awake.shape[:2]
    return {k: np.mean([
        wd_trials(v[:T_awake, g], awake[:,g]) for g in range(groups)
    ]) for (k,v) in replay_dict.items()}
'''