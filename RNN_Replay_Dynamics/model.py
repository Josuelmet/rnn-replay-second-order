import torch
from torch import nn

# TODO: allow for complex values
# When doing so, make sure to modify the leakage sigmoid stuff accordingly.

# TODO: the two papers apply noise differently.
#       The first applies noise outside of the ReLU, whereas the second applies it within the ReLU.
#       Which is better / more theoretically sound?? For langevin sampling??


class NoisyRNN(nn.Module):
    def __init__(self,
        d, hidden_dim,
        act=nn.Identity(),
        use_leak=True, use_norm=True,
        bias_in=True, bias_hidden=True, bias_out=True
    ):
        super(NoisyRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.act = act
        self.d = d

        self.W_in = nn.Linear(d, hidden_dim, bias=bias_in)
        self.W_out = nn.Linear(hidden_dim, d, bias=bias_out)
        self.W_hidden = nn.Linear(hidden_dim, hidden_dim, bias=bias_hidden)
        self.W_leak = nn.Parameter(torch.randn(hidden_dim)) if use_leak else None
        self.norm = nn.LayerNorm(hidden_dim) if use_norm else None

    def forward(self, h, x, sigma_r):

        preac_hx = self.W_hidden(h) + self.W_in(x)
        if self.norm:
            self.preac_hx = self.norm(preac_hx)
        delta_noise = torch.randn_like(h) * sigma_r

        if self.W_leak is None:
            next_h = self.act(preac_hx + delta_noise)
        else:
            next_h = torch.sigmoid(self.W_leak) * h + self.act(preac_hx + delta_noise)

        return next_h, self.W_out(next_h)


    def pinv(self, x):
        return (x - self.W_out.bias) @ torch.linalg.pinv(self.W_out.weight.data).T

    def range(self):
        A = self.W_out.weight.data
        #return A.T @ torch.linalg.inv(A @ A.T) @ A
        return A.T @ torch.linalg.pinv(A).T
    
    
    def sample(self, T, N, sigma_r,
               init_pos, intentions=None, observations=None,
               lambda_v=1, b_a=0, tau_a=50):
        
        if observations is None:
            observations = torch.zeros(T, N, self.d)

        r, v, c = [
            [torch.zeros(N, self.hidden_dim)] + [None] * (T-1) for _ in range(3)
        ]
        r[0] = self.pinv(init_pos)
        if intentions is not None:
            r[0] += intentions @ (torch.eye(self.hidden_dim) - self.range().T)
            
        for i in range(1, T):
            c[i] = c[i-1] + (1/tau_a) * (-c[i-1] + b_a * r[i-1]) # CHANGED FROM OLD VERSION
            next_r, _ = self.forward(r[i-1], observations[i], sigma_r)
            delta_r = next_r - r[i-1]
            v[i] = (1 - lambda_v) * v[i-1] + delta_r
            r[i] = r[i-1] + v[i] - c[i-1] 
            
        x_hat = self.W_out(torch.stack(r)) # shape = T, N, d
        return x_hat, r



# Ideally this function would go in the NoisyRNN class,
# but we made this after training all the models, so it will go outside.

def sample_plaus(
    self, T, N, sigma_r,
    init_pos, intentions=None, observations=None,
    lambda_v=1, b_a=0, tau_a=50 # CHANGE FROM SAMPLE
):

    if observations is None:
        observations = torch.zeros(T, N, self.d)

    r, c = [
        [torch.zeros(N, self.hidden_dim)] + [None] * (T-1) for _ in range(2)
    ]
    r[0] = self.pinv(init_pos)
    if intentions is not None:
        r[0] += intentions @ (torch.eye(self.hidden_dim) - self.range().T)
        
    for i in range(1, T):
        # NEW DEFINITION FOR ADAPTATION C
        c[i] = c[i-1] + (1/tau_a) * (-c[i-1] + b_a * r[i-1])
        if i >= 3:
            r_in = r[i-1] + (1 - lambda_v) * (r[i-2] - r[i-3])
        else:
            r_in = r[i-1]
        r[i] = self.forward(r_in, observations[i], sigma_r)[0] - c[i-1]
        
    x_hat = self.W_out(torch.stack(r)) # shape = T, N, d
    return x_hat, r
    
 





