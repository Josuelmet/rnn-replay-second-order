import numpy as np


def triangle_vertices_and_mus():
    tvs = np.array([[0,0], [1,0], [1/2, np.sqrt(3)/2]]) # starting points
    tmus = np.array([tvs[1], tvs[2], tvs[0], tvs[2], tvs[0], tvs[1]]) # ending points
    return tvs, tmus


def triangle(
    T=100, N=120,
    dt=0.02, leak=2,
    sigma_s=1, sigma_init=0.02,
    seed=None
):
    
    rng = np.random.default_rng(seed=seed)
    vertices, mus = triangle_vertices_and_mus() 
    groups, d = 6, 2
    s = np.zeros((T, groups, N//groups, 2))

    # Ornstein-Uhlenbeck processes
    s[0] = rng.normal(size=s[0].shape) * sigma_init # add starting noise if sigma_init > 0
    for i in range(groups):
        s[0, i] += vertices[i//2]
        for t in range(1,T):
            s[t,i] = s[t-1,i] + dt * (leak * (mus[i] - s[t-1,i]) + sigma_s * rng.normal(size=s[t,i].shape))
            
    return s




def tmaze_mus_t(T=90):
    T1, T2, T3 = int(T/3), int(T*2/3), T

    # steadily moving trajectory across [0, 1) in T1 timesteps
    linspace = np.arange(T1) / T1

    # construct two time-varying means (routes) that together form a T shape
    groups, d = 2, 2
    mus_t = np.zeros((T, groups, d)) # T timesteps, groups (2) routes, d (2) dimensions
    mus_t[:T1, 0, 1] = linspace   # route 0 first moves along dimension 1 (y)
    mus_t[T1:, 0, 1] = 1          # route 0 then stops moving along y,
    mus_t[T1:T2, 0, 0] = linspace # and moves along dimension 0 (x).
    mus_t[T2:T3, 0, 0] = 1        # then route 0 stops moving, settling at (1, 1).
    # route 1 is a copy of route 2,
    # except that it moves and rests in the opposite direction along dimension 0 (x)
    mus_t[:, 1] = mus_t[:,0]
    mus_t[:, 1, 0] *= -1
    return mus_t


def tmaze(
    T=90, N=120,
    leak=0.5,
    sigma_s=0.05, sigma_init=0.05,
    seed=None
):
    
    rng = np.random.default_rng(seed=seed)
    routes = tmaze_mus_t(T)
    groups, d = 2, 2

    # states have T timesteps, 2 groups of size N//2, and d (2) dimensions
    s = np.zeros((T, groups, N//groups, d))
    s[0] = rng.normal(size=s[0].shape) * sigma_init

    # Ornstein-Uhlenbeck processes with time-varying mean
    for i in range(groups): # group idx
        for t in range(1, T):
            s[t, i] = s[t-1, i] + leak * (routes[t, i] - s[t-1, i]) + sigma_s * rng.normal(size=s[t, i].shape)

    return s