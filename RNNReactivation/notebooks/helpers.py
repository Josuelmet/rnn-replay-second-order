import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

import torch

sys.path.append("..")
import utils

    
    
def optimal_decoder(model, observations, lr=1e-2, iters=31, verbose=False):
    out = torch.nn.Parameter(model.task.place_cells.decode_pos(observations.detach()), requires_grad=True)
    optim = torch.optim.Adam([out], lr=lr)
    for iteration in range(iters):
        loss = (model.task.place_cells.get_activation(out) - observations.detach()).square().mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        if verbose and iteration % 5 == 0:
            print(iteration, loss.item())
    return out.detach()


def gather_models(seed_dirs, epoch):
    models = []
    for seed_path in seed_dirs:
        seed = int(str(seed_path).split('_')[-1])
        utils.set_random_seeds(seed)
        
        model = torch.load(seed_path / f"model_{epoch}.pt", map_location="cpu")
        model.set_device("cpu")
        model.eval() 
        model.seed = seed
        models.append(model)
    return models


def gather_test_data(models):
    test_pos, test_obs, test_inits = [], [], []
    for model in models:
        utils.set_random_seeds(model.seed)
        test_batch = model.task.get_test_batch()
        test_pos.append(test_batch['target_pos'])
        test_obs.append(test_batch['targets'])
        test_inits.append(test_batch['init_state'])
    # shapes are seeds, N, T, dim
    return torch.stack(test_obs), torch.stack(test_pos), torch.stack(test_inits)


# NEW
# modified forward() function from https://github.com/nandahkrishna/RNNReactivation/blob/master/models/rnn.py
def momentum(self, lambda_v: float, tau_a: float, b_a: float,
                   quiescence: str, lambda_scaling: bool, x: torch.Tensor, init_state=None):
    batch_size, timesteps = x.shape[0], x.shape[1]

    if init_state is not None:
        self.h = self.encoder(init_state.reshape(batch_size, self.n_init))
    else:
        self.h = torch.zeros(batch_size, self.n_rec, device=self.device)
    self.v, self.c = torch.zeros_like(self.h), torch.zeros_like(self.h)

    self.h_1t = torch.zeros(batch_size, timesteps, self.n_rec, device=self.device)
    self.v_1t, self.c_1t = torch.zeros_like(self.h_1t), torch.zeros_like(self.h_1t)
    self.y_1t = torch.zeros(batch_size, timesteps, self.n_out, device=self.device)

    for t in range(timesteps):
        x_t = x[:, t, :].reshape(batch_size, self.n_in)
        noise_in = torch.rand_like(x_t, device=self.device)
        noise_rec = torch.rand_like(self.h, device=self.device)
        noise_out = torch.randn(batch_size, self.n_out, device=self.device)

        prev_c = self.c.clone()
        self.c += (1 / tau_a) * (-self.c + b_a * self.h)
        self.u = self.w_rec(self.h) + self.w_in(x_t + self.sigma_in * noise_in)
        next_h = (
            (1 - self.dt / self.tau) * self.h
            + (self.dt / self.tau) * self.activation(self.u)
            + self.sigma_rec * noise_rec * \
              np.sqrt(2 if quiescence == 'scaled' else 1) * \
              np.sqrt(lambda_v if lambda_scaling else 1)
        )
        delta_h = next_h - self.h
        self.v = (1 - lambda_v) * self.v + delta_h
        self.h += self.v - prev_c
        
        self.y = self.w_out(self.h) + self.sigma_out * noise_out

        self.h_1t[:, t, :] = self.h
        self.v_1t[:, t, :] = self.v
        self.c_1t[:, t, :] = self.c
        self.y_1t[:, t, :] = self.y

    return self.h_1t, self.y_1t


# NEW
def momentum_plaus(self, lambda_v: float, tau_a: float, b_a: float,
                         quiescence: str, lambda_scaling: bool, x: torch.Tensor, init_state=None):
    batch_size, timesteps = x.shape[0], x.shape[1]

    if init_state is not None:
        self.h = self.encoder(init_state.reshape(batch_size, self.n_init))
    else:
        self.h = torch.zeros(batch_size, self.n_rec, device=self.device)
    self.c = torch.zeros_like(self.h)

    self.h_1t = torch.zeros(batch_size, timesteps, self.n_rec, device=self.device)
    self.c_1t = torch.zeros_like(self.h_1t)
    self.y_1t = torch.zeros(batch_size, timesteps, self.n_out, device=self.device)

    for t in range(timesteps):
        x_t = x[:, t, :].reshape(batch_size, self.n_in)
        noise_in = torch.rand_like(x_t, device=self.device)
        noise_rec = torch.rand_like(self.h, device=self.device)
        noise_out = torch.randn(batch_size, self.n_out, device=self.device)
        
        prev_c = self.c.clone()
        self.c += (1 / tau_a) * (-self.c + b_a * self.h)
        if t >= 3:
            # coming into this line, self.h should equal self.h_1t[:, t-1]
            self.h += (1 - lambda_v) * (self.h_1t[:, t-2] - self.h_1t[:, t-3])
        self.u = self.w_rec(self.h) + self.w_in(x_t + self.sigma_in * noise_in)
        self.h = (
            (1 - self.dt / self.tau) * self.h
            + (self.dt / self.tau) * self.activation(self.u)
            + self.sigma_rec * noise_rec * \
              np.sqrt(2 if quiescence == 'scaled' else 1) * \
              np.sqrt(lambda_v if lambda_scaling else 1)
            - prev_c
        )
        self.y = self.w_out(self.h) + self.sigma_out * noise_out

        self.h_1t[:, t, :] = self.h
        self.c_1t[:, t, :] = self.c
        self.y_1t[:, t, :] = self.y

    return self.h_1t, self.y_1t


"""
# OLD
# modified forward() function from https://github.com/nandahkrishna/RNNReactivation/blob/master/models/rnn.py
def momentum(self, lambda_v: float, tau_a: float, b_a: float,
                   quiescence: str, lambda_scaling: bool, x: torch.Tensor, init_state=None):
    batch_size, timesteps = x.shape[0], x.shape[1]

    if init_state is not None:
        self.h = self.encoder(init_state.reshape(batch_size, self.n_init))
    else:
        self.h = torch.zeros(batch_size, self.n_rec, device=self.device)
    self.v, self.c = torch.zeros_like(self.h), torch.zeros_like(self.h)

    self.h_1t = torch.zeros(batch_size, timesteps, self.n_rec, device=self.device)
    self.v_1t, self.c_1t = torch.zeros_like(self.h_1t), torch.zeros_like(self.h_1t)
    self.y_1t = torch.zeros(batch_size, timesteps, self.n_out, device=self.device)

    for t in range(timesteps):
        x_t = x[:, t, :].reshape(batch_size, self.n_in)
        noise_in = torch.rand_like(x_t, device=self.device)
        self.u = self.w_rec(self.h) + self.w_in(x_t + self.sigma_in * noise_in)

        noise_rec = torch.rand_like(self.h, device=self.device)
        next_h = (
            (1 - self.dt / self.tau) * self.h
            + (self.dt / self.tau) * self.activation(self.u)
            + self.sigma_rec * noise_rec * \
              np.sqrt(2 if quiescence == 'scaled' else 1) * \
              np.sqrt(lambda_v if lambda_scaling else 1)
        )
        #delta_h = next_h - self.h - self.c
        delta_h = next_h - self.h
        #self.c = (1 / tau_a) * (self.c + b_a * self.h)
        self.v = (1 - lambda_v) * self.v + delta_h
        #self.h += self.v
        self.h += self.v - self.c
        self.c = (1 / tau_a) * (self.c + b_a * self.h)
        
        noise_out = torch.randn(batch_size, self.n_out, device=self.device)
        self.y = self.w_out(self.h) + self.sigma_out * noise_out

        self.h_1t[:, t, :] = self.h
        self.v_1t[:, t, :] = self.v
        self.c_1t[:, t, :] = self.c
        self.y_1t[:, t, :] = self.y

    return self.h_1t, self.y_1t
    
# OLD
def momentum_plaus(self, lambda_v: float, tau_a: float, b_a: float,
                         quiescence: str, lambda_scaling: bool, x: torch.Tensor, init_state=None):
    batch_size, timesteps = x.shape[0], x.shape[1]

    if init_state is not None:
        self.h = self.encoder(init_state.reshape(batch_size, self.n_init))
    else:
        self.h = torch.zeros(batch_size, self.n_rec, device=self.device)
    self.c = torch.zeros_like(self.h)

    self.h_1t = torch.zeros(batch_size, timesteps, self.n_rec, device=self.device)
    self.c_1t = torch.zeros_like(self.h_1t)
    self.y_1t = torch.zeros(batch_size, timesteps, self.n_out, device=self.device)

    for t in range(timesteps):
        x_t = x[:, t, :].reshape(batch_size, self.n_in)
        noise_in = torch.rand_like(x_t, device=self.device)
        if t >= 3:
            # self.h should equal self.h_1t[:, t-1] before this line
            self.h += (1 - lambda_v) * (self.h_1t[:, t-2] - self.h_1t[:, t-3])
        self.u = self.w_rec(self.h) + self.w_in(x_t + self.sigma_in * noise_in)

        noise_rec = torch.rand_like(self.h, device=self.device)
        next_h = (
            (1 - self.dt / self.tau) * self.h
            + (self.dt / self.tau) * self.activation(self.u)
            + self.sigma_rec * noise_rec * \
              np.sqrt(2 if quiescence == 'scaled' else 1) * \
              np.sqrt(lambda_v if lambda_scaling else 1)
        )
        self.h = next_h - self.c
        self.c = (1 / tau_a) * (self.c + b_a * self.h)
        
        noise_out = torch.randn(batch_size, self.n_out, device=self.device)
        self.y = self.w_out(self.h) + self.sigma_out * noise_out

        self.h_1t[:, t, :] = self.h
        self.c_1t[:, t, :] = self.c
        self.y_1t[:, t, :] = self.y

    return self.h_1t, self.y_1t
""";




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


def wd_types_to_csvs(wds_T, T, experiment, col_param, root_path=None):
    assert experiment in ['unbiased', 'biased']
    wd_dfs = {}
    for wd_key, wd_set in wds_T.items():
        wd_means = {k: v.mean() for k,v in wd_set.items()}
        path = None if root_path is None else \
               os.path.join(root_path, f'wds_{T}_{experiment}_{wd_key}.csv')
        wd_dfs[wd_key] = dict_to_csv(wd_means, col_param, path)
    return wd_dfs





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