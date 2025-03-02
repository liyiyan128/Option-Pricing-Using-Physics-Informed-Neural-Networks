import torch
# import numpy as np
from matplotlib import pyplot as plt
from PINN.utilities import collocation_points


class VanillaOptionPINN(torch.nn.Module):
    def __init__(self, nn, K, T, r, sigma, S_inf=None,
                 type='put', style='european',
                 device=None):
        super(VanillaOptionPINN, self).__init__()
        self.loss_history = {
            'ib': [],
            'pde': [],
            'data': [],
            'total': [],
            'valid': [],
        }
        self.type = type.lower()
        self.style = style.lower()
        self.device = device or 'cpu'
        self.V_nn = nn.to(self.device)
        self.register_buffer('K', torch.tensor(K))
        self.register_buffer('T', torch.tensor(T))
        self.register_buffer('r', torch.tensor(r))
        self.register_buffer('sigma', torch.tensor(sigma))
        self.register_buffer('S_inf', torch.tensor(S_inf) if S_inf is not None else torch.tensor(3*K))
        self.to(self.device)

    def forward(self, tau, S):
        V = self.V_nn(torch.cat((tau, S), dim=1))
        return V

    def pde_nn(self, tau, S):
        V = self.forward(tau, S)
        V_tau = torch.autograd.grad(V, tau, grad_outputs=torch.ones_like(V),
                                    create_graph=True, retain_graph=True, allow_unused=True)[0]
        V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                                  create_graph=True, retain_graph=True, allow_unused=True)[0]
        V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                                   create_graph=True, retain_graph=True, allow_unused=True)[0]
        pde = V_tau - 0.5*self.sigma**2*S**2*V_SS - self.r*S*V_S + self.r*V
        
        if self.style == 'european':
            return pde
        elif self.style == 'american':
            intrinsic = torch.where(torch.tensor(self.type == 'put'),
                                    torch.maximum(self.K - S, torch.zeros_like(S)),
                                    torch.maximum(S - self.K, torch.zeros_like(S)))
            return torch.min(pde, V-intrinsic)  # linear complementarity condition

    def loss_ib(self, tau, S, V):
        if tau is None:
            return torch.tensor(0.)
        return torch.mean((self.forward(tau, S) - V)**2)

    def loss_pde(self, tau, S):
        if tau is None:
            return torch.tensor(0.)
        pde = self.pde_nn(tau, S)
        if pde is None:
            raise ValueError('pde is None')
        return torch.mean(self.pde_nn(tau, S)**2)

    def loss_data(self, tau, S, V):
        if tau is None:
            return torch.tensor(0.)
        return torch.mean((self.forward(tau, S) - V)**2)

    def loss(self,
             tau_ib=None, S_ib=None, V_ib=None,
             tau_pde=None, S_pde=None,
             tau_data=None, S_data=None, V_data=None,
             return_tensor=True):
        if return_tensor:
            return self.loss_ib(tau_ib, S_ib, V_ib), self.loss_pde(tau_pde, S_pde), self.loss_data(tau_data, S_data, V_data)
        else:
            return self.loss_ib(tau_ib, S_ib, V_ib).detach().item(), self.loss_pde(tau_pde, S_pde).detach().item(), self.loss_data(tau_data, S_data, V_data).detach().item()

    def predict(self, tau, S):
        return self.forward(tau, S)

    def fit(self,
            tau_ib, S_ib, V_ib,
            tau_pde=None, S_pde=None,
            tau_data=None, S_data=None, V_data=None,
            valid=False, valid_grid=(100, 100), tau_valid=None, S_valid=None,
            N_pde=2000, sampling='sobol', resample=False,
            loss_weights=(1., 1., 1.),
            epochs=200, optimizer='adam', verbose=True, **kwargs):

        if tau_pde is None or S_pde is None:
            # initialise sobol engine for sobol/adaptive sampling
            print(f'No collocation points provided\nSampling {N_pde} collocation points ({sampling})')
            sobol = torch.quasirandom.SobolEngine(dimension=2) if (sampling == 'sobol' or sampling == 'adaptive') else None
            tau_pde, S_pde = collocation_points(self, N_pde, sampling=sampling, sobol=sobol, **kwargs)

        if valid:
            if tau_valid is None or S_valid is None:
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
        print(f'On device: {self.device}')
        
        resample = int(resample)

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), **kwargs)
        elif optimizer == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(self.parameters(), **kwargs)   
        print(f'Optimizer: {optimizer}')

        def closure():
            self.optimizer.zero_grad()
            loss_ib, loss_pde, loss_data = self.loss(tau_ib, S_ib, V_ib, tau_pde, S_pde, tau_data, S_data, V_data)
            loss = loss_weights[0]*loss_ib + loss_weights[1]*loss_pde + loss_weights[2]*loss_data
            loss.backward(retain_graph=True)
            return loss

        # Training loop
        for i in range(epochs):

            if resample and (i+1) % resample == 0:
                S_pde_base = S_pde.clone() if sampling == 'adaptive' else None
                tau_pde_base = tau_pde.clone() if sampling == 'adaptive' else None
                tau_pde, S_pde = collocation_points(self, N_pde, sampling=sampling, sobol=sobol,
                                                    tau_pde_base=tau_pde_base, S_pde_base=S_pde_base, **kwargs)
                S_pde, tau_pde = S_pde.to(self.device), tau_pde.to(self.device)

            self.optimizer.step(closure)

            loss_ib, loss_pde, loss_data = self.loss(tau_ib, S_ib, V_ib, tau_pde, S_pde, tau_data, S_data, V_data, return_tensor=False)
            total = loss_ib + loss_pde + loss_data
            self.loss_history['ib'].append(loss_ib)
            self.loss_history['pde'].append(loss_pde)
            if tau_data is not None:
                self.loss_history['data'].append(loss_data)
            self.loss_history['total'].append(total)
            if valid:
                loss_valid = self.loss_pde(tau_valid, S_valid).detach().item()
                self.loss_history['valid'].append(loss_valid)
            if verbose and (i+1) % 100 == 0:
                print(f'Epoch {i+1}/{epochs}    |    Loss: {total}')

    def plot_history(self, ib=True, pde=True, data=True, valid=True, range=-1, log_scale=True,
                  title='Loss History', figsize=(10, 8), fontsize=16,
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
