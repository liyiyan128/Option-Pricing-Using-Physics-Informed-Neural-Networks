import torch
# import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from PINN.utilities import collocation_points


class EuropeanPINN(torch.nn.Module):
    def __init__(self, nn, K, T, r, sigma, S_inf=None, type='put', device=None):
        super(EuropeanPINN, self).__init__()
        self.loss_history = {
            'ib': [],
            'pde': [],
            'data': [],
            'total': [],
            'valid': [],
        }
        self.type = type
        self.device = device or 'cpu'
        self.V_nn = nn.to(self.device)
        self.register_buffer('K', torch.tensor(K))
        self.register_buffer('T', torch.tensor(T))
        self.register_buffer('r', torch.tensor(r))
        self.register_buffer('sigma', torch.tensor(sigma))
        self.register_buffer('S_inf', torch.tensor(S_inf) if S_inf is not None else torch.tensor(3*K))
        self.to(self.device)

    def forward(self, S, tau):
        V = self.V_nn(torch.cat((S, tau), dim=1))
        return V

    def pde_nn(self, S, tau):
        V = self.forward(S, tau)
        V_tau = torch.autograd.grad(V, tau, grad_outputs=torch.ones_like(V),
                                    create_graph=True, retain_graph=True, allow_unused=True)[0]
        V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                                  create_graph=True, retain_graph=True, allow_unused=True)[0]
        V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                                   create_graph=True, retain_graph=True, allow_unused=True)[0]

        return V_tau - 0.5*self.sigma**2*S**2*V_SS - self.r*S*V_S + self.r*V

    def loss_ib(self, S, tau, V):
        if S is None:
            return torch.tensor(0.)
        return torch.mean((self.forward(S, tau) - V)**2)

    def loss_pde(self, S, tau):
        if S is None:
            return torch.tensor(0.)
        return torch.mean(self.pde_nn(S, tau)**2)

    def loss_data(self, S, tau, V):
        if S is None:
            return torch.tensor(0.)
        return torch.mean((self.forward(S, tau) - V)**2)

    def loss(self,
             S_ib=None, tau_ib=None, V_ib=None,
             S_pde=None, tau_pde=None,
             S_data=None, tau_data=None, V_data=None,
             return_tensor=True):
        # return self.loss_ib(S_ib, tau_ib, V_ib) + \
        #        self.loss_pde(S_pde, tau_pde) + \
        #        self.loss_data(S_data, tau_data, V_data):
        if return_tensor:
            return self.loss_ib(S_ib, tau_ib, V_ib), self.loss_pde(S_pde, tau_pde), self.loss_data(S_data, tau_data, V_data)
        else:
            return self.loss_ib(S_ib, tau_ib, V_ib).detach().item(), self.loss_pde(S_pde, tau_pde).detach().item(), self.loss_data(S_data, tau_data, V_data).detach().item()

    def predict(self, x):
        return self.forward(x)

    def fit(self, S_ib, tau_ib, V_ib,
            S_pde=None, tau_pde=None,
            S_data=None, tau_data=None, V_data=None,
            valid=False, valid_grid=(1000, 1000), S_valid=None, tau_valid=None,
            N_pde=2000, resample=False, sampling='uniform',
            loss_weights=(1., 1., 1.),
            epochs=1000, optimizer='adam', verbose=True, **kwargs):

        if S_pde is None or tau_pde is None:
            sobol = torch.quasirandom.SobolEngine(dimension=2) if (sampling == 'sobol' or sampling == 'adaptive') else None
            S_pde, tau_pde = collocation_points(N_pde, self, sampling=sampling, sobol=sobol, **kwargs)

        if valid:
            if S_valid is None or tau_valid is None:
                S_valid = torch.linspace(0, self.S_inf, valid_grid[1])
                tau_valid = torch.linspace(0, self.T, valid_grid[0])
                S_valid, tau_valid = torch.meshgrid(S_valid, tau_valid, indexing='xy')
            S_valid = S_valid.reshape(-1, 1).requires_grad_(True)
            tau_valid = tau_valid.reshape(-1, 1).requires_grad_(True)
    
        S_ib, tau_ib, V_ib, S_pde, tau_pde, S_data, tau_data, V_data, S_valid, tau_valid = \
            S_ib.to(self.device), tau_ib.to(self.device), V_ib.to(self.device), \
            S_pde.to(self.device), tau_pde.to(self.device), \
            S_data.to(self.device) if S_data is not None else None, \
            tau_data.to(self.device) if tau_data is not None else None, \
            V_data.to(self.device) if V_data is not None else None, \
            S_valid.to(self.device) if S_valid is not None else None, \
            tau_valid.to(self.device) if tau_valid is not None else None
        
        resample = int(resample)

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), **kwargs)
        elif optimizer == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(self.parameters(), **kwargs)   

        def closure():
            self.optimizer.zero_grad()
            loss_ib, loss_pde, loss_data = self.loss(S_ib, tau_ib, V_ib, S_pde, tau_pde, S_data, tau_data, V_data)
            loss = loss_weights[0]*loss_ib + loss_weights[1]*loss_pde + loss_weights[2]*loss_data
            loss.backward(retain_graph=True)
            return loss

        # Training loop
        for i in range(epochs):

            if resample and (i+1) % resample == 0:
                S_pde_base = S_pde.clone() if sampling == 'adaptive' else None
                tau_pde_base = tau_pde.clone() if sampling == 'adaptive' else None
                S_pde, tau_pde = collocation_points(N_pde, self, sampling=sampling, sobol=sobol,
                                                    S_pde_base=S_pde_base, tau_pde_base=tau_pde_base, **kwargs)
                S_pde, tau_pde = S_pde.to(self.device), tau_pde.to(self.device)

            self.optimizer.step(closure)

            loss_ib, loss_pde, loss_data = self.loss(S_ib, tau_ib, V_ib, S_pde, tau_pde, S_data, tau_data, V_data, return_tensor=False)
            total = loss_ib + loss_pde + loss_data
            self.loss_history['ib'].append(loss_ib)
            self.loss_history['pde'].append(loss_pde)
            if S_data is not None:
                self.loss_history['data'].append(loss_data)
            self.loss_history['total'].append(total)
            if valid:
                _, loss_pde, _ = self.loss(S_pde=S_valid, tau_pde=tau_valid, return_tensor=False)
                self.loss_history['valid'].append(loss_pde)
            if verbose and (i+1) % 100 == 0:
                print(f'Epoch {i+1}/{epochs}:\nIB Loss: {loss_ib}\nPDE Loss: {loss_pde}\nData Loss: {loss_data}\n')

    def plot_loss(self, ib=True, pde=True, data=True, valid=True,
                  range=-1, log_scale=True, title='Loss History', figsize=(10, 8), fontsize=16,
                  save=False, file_name='loss_history.pdf'):
        plt.figure(figsize=figsize)
        plt.plot(self.loss_history['total'][:range], label='Total loss', c='red')
        if ib:
            plt.plot(self.loss_history['ib'][:range], label='IB loss ($MSE_B$)', ls='--', alpha=0.8)
        if pde:
            plt.plot(self.loss_history['pde'][:range], label='PDE loss ($MSE_F$)', ls='--', alpha=0.8)
        if data and len(self.loss_history['data']) > 0:
            plt.plot(self.loss_history['data'][:range], label='Data loss ($MSE_{data}$)', ls='--', alpha=0.8)
        if valid and len(self.loss_history['valid']) > 0:
            plt.plot(self.loss_history['valid'][:range], label='Validation loss', ls='--', alpha=0.8)
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Loss', fontsize=fontsize)
        if log_scale:
            plt.yscale('log')
        plt.title(title, fontsize=fontsize+2)
        plt.legend(fontsize=fontsize+2)
        plt.tight_layout()
        if save:
            plt.savefig(file_name, bbox_inches='tight')
        plt.show()
