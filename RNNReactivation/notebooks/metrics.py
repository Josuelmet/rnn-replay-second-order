import numpy as np
import scipy.stats as stats
import sys
from tqdm import tqdm

import ot

sys.path.append("..")
import utils



def wasserstein(x, y, n_projections=100):
    return ot.sliced_wasserstein_distance(x, y, n_projections=n_projections).item()

def calc_wds(replay, test_pos, n_projections=10_000):
    # treating time as a dimension
    assert test_pos.ndim == 4 and test_pos.shape[-1] == 2
    wds = {}
    t_lim = test_pos.shape[2] # seeds, N, T
    for (k,v) in tqdm(replay.items()):
        # each v has shape (seeds, N, T, 2)
        wds[k] = wasserstein(test_pos.flatten(0,1).flatten(1),
                             v[:,:,:t_lim].flatten(0,1).flatten(1),
                             n_projections=n_projections)
    return wds

'''
def calc_wds(replay_xy, test_pos, t_lim, n_projections=100):
    wds = {}
    for (k,v) in tqdm(replay_xy.items()):
        wds[k] = []
        for i in range(len(test_pos)):
            wds[k].append(wasserstein(test_pos[i,:,:t_lim].flatten(0,1),
                                      v[i,:,:t_lim].flatten(0,1), n_projections=n_projections))
    return wds
'''

'''
def calc_wds(replay_xy, test_pos, t_lim, timewise, seedwise, n_projections=100):
    wds = {}
    assert (test_pos.shape[2] <= t_lim) or not timewise
    for (k,v) in tqdm(replay_xy.items()):
        # each v has shape (seeds, N, T, 2)
        wds[k] = []
        add_wd = lambda x,y: wds[k].append(wasserstein(x, y, n_projections=n_projections))
        if seedwise:
            for seed in range(len(test_pos)):
                seed_test = test_pos[seed,:,:t_lim] # N, t_lim, 2
                seed_v = v[seed,:,:t_lim]
                if timewise:
                    for t in range(t_lim):
                        add_wd(seed_test[:,t], seed_v[:,t])
                else:
                    add_wd(seed_test.flatten(0,1), seed_v.flatten(0,1))
        elif timewise:
            for t in range(t_lim):
                add_wd(test_pos[:,:,t].flatten(0,1), v[:,:,t].flatten(0,1))
        else:
            add_wd(test_pos[:,:,:t_lim].reshape(-1,2), v[:,:,:t_lim].reshape(-1,2))
        wds[k] = np.array(wds[k])
    return wds
                                   
        


def calc_kls(replay_xy, test_pos, t_lim):
    kls = {}
    kde_plots = {}

    xmin, xmax = -1.25, 1.25
    ymin, ymax = -1.25, 1.25
    kernels_test = [stats.gaussian_kde(tp.reshape(-1,2).T) for tp in test_pos]
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    for (k,v) in tqdm(replay_xy.items()):
        kls[k], kde_plots[k] = [], []
        for i in range(len(test_pos)):
            kernel = stats.gaussian_kde(v[i,:,:t_lim].reshape(-1,2).T)
            kls[k].append(utils.kl_divergence(kernels_test[i], kernel, n=5_000))
            kde_plots[k].append(np.reshape(kernel(positions).T, xx.shape))
    return kls, kde_plots
'''