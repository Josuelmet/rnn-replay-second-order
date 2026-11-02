import numpy as np
import pandas as pd

import datasets



def assign_regions(dataset, replay_groupseed):
    assert replay_groupseed.ndim == 3 and replay_groupseed.shape[-1] == 2
    assert dataset in ['tmaze', 'triangle']
    if dataset == 'tmaze':
        vertices = datasets.tmaze_mus_t()[-1] # [[-1,1], [1,1]]. shape = group, 2
        #vertices = np.vstack([np.zeros((1,2)), vertices]) # add a [0,0] vertex (starting point)
        vertices = np.vstack([vertices.mean(0), vertices]) # add a [0,1] vertex (middle point)
    elif dataset == 'triangle':
        vertices, _ = datasets.triangle_vertices_and_mus()
    # replay_set shape = T, N, 2
    dists = np.stack([replay_groupseed - v for v in vertices]) # 3, T, N, 2
    dists = np.sqrt((dists**2).sum(-1)) # distance to each vertex
    return dists.argmin(0) # assign closest vertex

def smooth_regions(region_list, min_duration=10):
    r = region_list.copy()
    change_idxs = np.where(np.diff(r) != 0)[0] + 1
    change_idxs = np.concatenate(([0], change_idxs, [len(r)]))
    val = r[0]
    for i1, i2 in zip(change_idxs[:-1], change_idxs[1:]):
        assert np.all(r[i1:i2] == r[i1])
        if i2 - i1 < min_duration:
            r[i1:i2] = val
        else:
            val = r[i1]
    return r
# simple test
assert all(smooth_regions(np.arange(10), min_duration=1) == np.arange(10))

def region_counts(dataset, replay_groupseed, min_duration=10):
    # replay_groupseed shape = T, N, 2
    regions = assign_regions(dataset, replay_groupseed) # T, N
    for j in range(regions.shape[1]):
        regions[:, j] = smooth_regions(regions[:, j], min_duration)
    # number of region changes + 1
    region_counts = (np.diff(regions, axis=0) != 0).sum(0) + 1
    return regions, region_counts




# plotting region assignments by groupseed over time
def plot_region_assignments_over_time(dataset, axs, replay_set,
                                      seeds_per_group=None, min_duration=10):
    # replay_set shape = T, groups, seeds, N_per_group, 2    
    assert replay_set.ndim == 5 and replay_set.shape[-1] == 2
    groups = replay_set.shape[1]
    
    for g in range(groups):
        regions = []
        for s_i in range(replay_set.shape[2]) if seeds_per_group is None else seeds_per_group[g]:
            groupseed_replay = replay_set[:, g, s_i]
            # transpose to have time be dim 1
            regions.append(region_counts(dataset, groupseed_replay, min_duration)[0].T)
        regions = np.vstack(regions)
        mean_regions = (np.diff(regions, axis=1) != 0).sum(1).mean() + 1
        axs[g].imshow(regions, cmap='bwr', interpolation='nearest')
        axs[g].set(title=f'Group {g}: mean {mean_regions :.2f}', facecolor='gray')
        
        
        
        
        
        
        
def calc_exploration_metrics(dataset, replays, min_duration=10):

    groups, seeds = list(replays.values())[0].shape[1:3]

    # every_region_count_value[param_str] shape = (#groups, #seeds, #trials/group/seed)
    every_region_count_value = {k: np.zeros(list(replays.values())[0].shape[1:-1]) \
                                for k in replays.keys()}
    every_distance           = {k: np.zeros(list(replays.values())[0].shape[1:-1]) \
                                for k in replays.keys()}
    region_count_means       = {}
    distance_means           = {}
    
    for params, replay_set in replays.items():
        rcs, dists = [], []
        for g in range(groups):
            for seed in range(seeds):
                groupseed_replay = replay_set[:, g, seed]

                rc_vals = region_counts(dataset, groupseed_replay, min_duration)[1]
                every_region_count_value[params][g][seed] = rc_vals
                rcs.extend(rc_vals)

                distances = groupseed_replay.diff(axis=0).square().sum(-1).sqrt().sum(0).numpy()
                every_distance[params][g][seed] = distances
                dists.extend(distances)
        region_count_means[params] = np.mean(rcs)
        distance_means[params] = np.mean(dists)

    return every_region_count_value, region_count_means, \
           every_distance, distance_means




def region_counts_histogram(region_count_vals, min_count=2, max_count=9):
    if not (all(region_count_vals >= min_count) and all(region_count_vals <= max_count)):
        print(f"Warning: region counts {region_count_vals} are not within {[min_count, max_count]}")
    bins = np.arange(min_count, max_count+2) - 0.5
    return np.histogram(region_count_vals, bins=bins)[0] # returns counts along each bin


def region_counts_histogram_df(every_region_count_value, param_combo_list, col_param,
                               group, seed, min_count=2, max_count=9):
    cols = []
    histograms = np.zeros((max_count - min_count + 1, len(param_combo_list)))
    for p_i, params in enumerate(param_combo_list):
        histograms[:,p_i] = region_counts_histogram(
            every_region_count_value[str(params)][group,seed], min_count, max_count)
        cols.append(f"{col_param}={params[col_param]}")

    df = pd.DataFrame(histograms, columns=cols)
    df.index = np.arange(min_count, max_count+1)
    return df 