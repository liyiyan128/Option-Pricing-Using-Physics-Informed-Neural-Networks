import torch
# import numpy as np
from matplotlib import pyplot as plt
from PINN.utilities import collocation_points
from itertools import cycle


class VanillaOptionPINN(torch.nn.Module):
    """
    Physics-Informed Neural Network (PINN) for pricing vanilla options.

    Price European or American options using a neural network to approximate
    the option value function V(tau, S) satisfying the Black-Scholes PDE.

    Parameters
    ----------
    nn : torch.nn.Module
        Neural network architecture.
    K, T, sigma, r : float
        Option parameters: strike price, maturity, volatility, risk-free rate.
    S_inf : float, optional
        Upper bound to truncate the price domain. Default is 3*K.
    scale : bool, float, optional
        Scale the price domain to [0, 1] if True, or scale by the given value.
    type, style : str, optional
        Option type: 'call' or 'put', and style: 'european' or 'american'.
    """

    def __init__(self, nn, K, T, sigma, r,
                 S_inf=None, scale=True,
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
        self.register_buffer('sigma', torch.tensor(sigma))
        self.register_buffer('r', torch.tensor(r))
        self.register_buffer('S_inf', torch.tensor(S_inf)
                             if S_inf is not None else torch.tensor(3*K))
        if isinstance(scale, bool):
            scale = K if scale else 1
        self.register_buffer('scale', torch.tensor(scale))
        self.to(self.device)

    def forward(self, tau, S):
        S_scaled = S/self.scale
        V_scaled = self.V_nn(torch.cat((tau, S_scaled), dim=1))
        return V_scaled*self.scale  # to original scale

    def pde_nn(self, tau, S):
        S_scaled = S/self.scale
        V_scaled = self.V_nn(torch.cat((tau, S_scaled), dim=1))
        V_tau = torch.autograd.grad(V_scaled, tau,
                                    grad_outputs=torch.ones_like(V_scaled),
                                    create_graph=True, retain_graph=True, allow_unused=True)[0]
        V_S = torch.autograd.grad(V_scaled, S_scaled,
                                  grad_outputs=torch.ones_like(V_scaled),
                                  create_graph=True, retain_graph=True, allow_unused=True)[0]
        V_SS = torch.autograd.grad(V_S, S_scaled,
                                   grad_outputs=torch.ones_like(V_S),
                                   create_graph=True, retain_graph=True, allow_unused=True)[0]
        pde = V_tau - 0.5*self.sigma**2*S_scaled**2*V_SS - self.r*S_scaled*V_S + self.r*V_scaled
        if self.style == 'european':
            return pde
        elif self.style == 'american':
            K_scaled = self.K/self.scale
            intrinsic = torch.where(torch.tensor(self.type == 'put'),
                                    torch.maximum(K_scaled-S_scaled, torch.zeros_like(S_scaled)),
                                    torch.maximum(S_scaled-K_scaled, torch.zeros_like(S_scaled)))
            return torch.min(pde, V_scaled-intrinsic)  # linear complementarity condition

    def loss_ib(self, tau, S, V):
        if tau is None:
            return torch.tensor(0.)
        # ib = torch.mean(self.forward(tau, S) - V)**2
        S_scaled = S/self.scale
        V_scaled = V/self.scale
        V_nn = self.V_nn(torch.cat((tau, S_scaled), dim=1))
        ib = torch.mean((V_nn - V_scaled)**2)
        return ib

    def loss_pde(self, tau, S):
        if tau is None:
            return torch.tensor(0.)
        pde = self.pde_nn(tau, S)
        return torch.mean(pde**2)

    def loss_data(self, tau, S, V):
        if tau is None:
            return torch.tensor(0.)
        # return torch.mean((self.forward(tau, S) - V)**2)
        S_scaled = S/self.scale
        V_scaled = V/self.scale
        return torch.mean((self.V_nn(torch.cat((tau, S_scaled), dim=1))-V_scaled)**2)

    def loss(self,
             tau_ib=None, S_ib=None, V_ib=None,
             tau_pde=None, S_pde=None,
             tau_data=None, S_data=None, V_data=None,
             return_tensor=True):
        if return_tensor:
            return self.loss_ib(tau_ib, S_ib, V_ib), \
                   self.loss_pde(tau_pde, S_pde), \
                   self.loss_data(tau_data, S_data, V_data)
        else:
            return self.loss_ib(tau_ib, S_ib, V_ib).detach().item(), \
                   self.loss_pde(tau_pde, S_pde).detach().item(), \
                   self.loss_data(tau_data, S_data, V_data).detach().item()

    def fit(self,
            tau_ib, S_ib, V_ib,
            tau_pde=None, S_pde=None,
            tau_data=None, S_data=None, V_data=None,
            valid=False, valid_grid=(100, 100), tau_valid=None, S_valid=None,
            N_pde=2000, sampling='sobol', resample=0,
            epochs=200, optimizer='adam',
            ib_batch_size=-1, pde_batch_size=-1, data_batch_size=-1,
            loss_weights=(1., 1., 1.),
            verbose=True, **kwargs):
        """
        Parameters
        ----------
        tau_ib, S_ib, V_ib : torch.Tensor
            Initial/boundary data.
        tau_pde, S_pde : torch.Tensor, optional
            Additional PDE collocation points.
        tau_data, S_data, V_data : torch.Tensor, optional
            Observed data points.
        valid : bool, optional
            If True, validate the model every epoch
            on valid_grid or provided validation points (tau_valid, S_valid).
        N_pde : int, optional
            Number of collocation points to sample.
        sampling : str, optional
            Sampling method: 'sobol', 'uniform', 'adaptive'.
        resample : int, optional
            Resample collocation points every resample epochs.
        epochs : int, optional
            Number of training epochs.
        optimizer : str, optional
            Optimizer: 'adam', 'lbfgs'.
        ib_batch_size, pde_batch_size, data_batch_size : int, optional
            Batch size for initial/boundary data, PDE collocation points, and observed data.
        loss_weights : tuple, optional
            Weights for the IB, PDE, and data losses.
        kwargs : dict, optional
            Additional arguments for optimizer and learning rate scheduler.
        """

        # initialise sobol engine for sobol/adaptive sampling
        sobol = torch.quasirandom.SobolEngine(dimension=2) if (sampling == 'sobol' or sampling == 'adaptive') else None
        _tau, _S = collocation_points(self, N_pde,
                                      sampling=sampling if sampling != 'adaptive' else 'sobol',
                                      sobol=sobol, **kwargs)
        if tau_pde is None:
            tau_pde, S_pde = _tau, _S
        else:
            tau_pde = torch.cat((tau_pde, _tau), dim=0)
            S_pde = torch.cat((S_pde, _S), dim=0)

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

        ib_batch_size = ib_batch_size if ib_batch_size > 0 else len(tau_ib)
        pde_batch_size = pde_batch_size if pde_batch_size > 0 else len(tau_pde)
        shuffle = kwargs.pop('shuffle', False)
        ib_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tau_ib, S_ib, V_ib),
                                                batch_size=ib_batch_size, shuffle=shuffle)
        pde_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tau_pde, S_pde),
                                                 batch_size=pde_batch_size, shuffle=shuffle)
        if tau_data is not None:
            data_batch_size = data_batch_size if data_batch_size > 0 else len(tau_data)
            data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tau_data, S_data, V_data),
                                                      batch_size=data_batch_size, shuffle=shuffle)

        resample = int(resample)
        verbose = int(verbose)

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), kwargs.pop('lr', 1e-3))
        elif optimizer == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(self.parameters(),
                                               lr=kwargs.pop('lr', 1.0),
                                               max_iter=kwargs.pop('max_iter', 20),
                                               history_size=kwargs.pop('history_size', 100),
                                               line_search_fn=kwargs.pop('line_search_fn', 'strong_wolfe'))
        early_stopping = kwargs.pop('early_stopping', 10)
        scheduler = kwargs.pop('scheduler', None)
        if scheduler is not None:
            scheduler = scheduler(self.optimizer, **kwargs)

        if verbose:
            print(f'Device: {self.device}')
            print(f'Optimizer: {optimizer}')
        min_loss = float('inf')
        patience = 0
        for i in range(epochs):

            if resample and (i+1) % resample == 0:
                tau_pde, S_pde = collocation_points(self, N_pde,
                                                    sampling=sampling, sobol=sobol, **kwargs)
                S_pde, tau_pde = S_pde.to(self.device), tau_pde.to(self.device)
                pde_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tau_pde, S_pde),
                                                         batch_size=pde_batch_size, shuffle=shuffle)

            pde_iter = iter(pde_loader)
            ib_iter = iter(cycle(ib_loader))
            data_iter = iter(cycle(data_loader)) if tau_data is not None else None
            for tau_pde_batch, S_pde_batch in pde_iter:
                tau_ib_batch, S_ib_batch, V_ib_batch = next(ib_iter)
                tau_data_batch, S_data_batch, V_data_batch = next(data_iter) if tau_data is not None else (None, None, None)

                def closure():
                    self.optimizer.zero_grad()
                    loss_ib, loss_pde, loss_data = self.loss(tau_ib_batch, S_ib_batch, V_ib_batch,
                                                             tau_pde_batch, S_pde_batch,
                                                             tau_data_batch, S_data_batch, V_data_batch)
                    loss = loss_weights[0]*loss_ib + loss_weights[1]*loss_pde + loss_weights[2]*loss_data
                    loss.backward(retain_graph=True)
                    return loss

                loss = self.optimizer.step(closure)
                if torch.isnan(loss):
                    break
            if torch.isnan(loss):
                print(f'NaN loss detected at epoch {i+1}')
                break

            loss_ib, loss_pde, loss_data = self.loss(tau_ib, S_ib, V_ib,
                                                     tau_pde, S_pde,
                                                     tau_data, S_data, V_data,
                                                     return_tensor=False)
            total = loss_ib + loss_pde + loss_data
            self.loss_history['ib'].append(loss_ib)
            self.loss_history['pde'].append(loss_pde)
            if tau_data is not None:
                self.loss_history['data'].append(loss_data)
            self.loss_history['total'].append(total)
            if valid:
                loss_valid = self.loss_pde(tau_valid, S_valid).detach().item()
                self.loss_history['valid'].append(loss_valid)
            if verbose and (i+1) % verbose == 0:
                print(f'Epoch {i+1}/{epochs}    |    Loss: {total}')

            if total >= min_loss:
                patience += 1
                if patience >= early_stopping:
                    if verbose:
                        print(f'Early stopping at epoch {i+1}')
                    break
            else:
                min_loss = total
                patience = 0

            if scheduler is not None:
                if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    scheduler.step(total)
                else:
                    scheduler.step()

    def plot_history(self, ib=True, pde=True, data=True, valid=True,
                     range=-1, log_scale=True,
                     title='Loss History', figsize=(10, 8), fontsize=16,
                     save=False, file_name='loss_history.pdf'):
        """Plot the loss history."""
        plt.figure(figsize=figsize)
        plt.plot(self.loss_history['total'][:range],
                 label='Total loss', c='red')
        if ib:
            plt.plot(self.loss_history['ib'][:range],
                     label='IB loss ($MSE_B$)', ls='--', alpha=0.8)
        if pde:
            plt.plot(self.loss_history['pde'][:range],
                     label='PDE loss ($MSE_F$)', ls='--', alpha=0.8)
        if data and len(self.loss_history['data']) > 0:
            plt.plot(self.loss_history['data'][:range],
                     label='Data loss ($MSE_{data}$)', ls='--', alpha=0.8)
        if valid and len(self.loss_history['valid']) > 0:
            plt.plot(self.loss_history['valid'][:range],
                     label='Validation loss', c='k', alpha=0.8)
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

    def evaluate_greeks(self, tau, S, greeks='delta'):
        """Compute the option greeks.

        Supported greeks: 'delta', 'theta', 'gamma', 'charm', 'speed', 'color'.
        """
        greeks = greeks.lower()
        V = self.forward(tau, S)

        if greeks == 'delta':
            V_S = torch.autograd.grad(V, S,
                                      grad_outputs=torch.ones_like(V),
                                      create_graph=True, retain_graph=True, allow_unused=True)[0]
            delta = V_S
            return delta
        elif greeks == 'theta' or greeks == 'time_decay':
            V_tau = torch.autograd.grad(V, tau,
                                        grad_outputs=torch.ones_like(V),
                                        create_graph=True, retain_graph=True, allow_unused=True)[0]
            theta = -V_tau
            return theta
        elif greeks == 'gamma':
            V_S = torch.autograd.grad(V, S,
                                      grad_outputs=torch.ones_like(V),
                                      create_graph=True, retain_graph=True, allow_unused=True)[0]
            V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                                       create_graph=True, retain_graph=True, allow_unused=True)[0]
            gamma = V_SS
            return gamma
        elif greeks == 'charm' or greeks == 'delta_decay':
            # charm = -dDelta/dtau = dTheta/dS = -d^2V/dtaudS
            V_S = torch.autograd.grad(V, S,
                                      grad_outputs=torch.ones_like(V),
                                      create_graph=True, retain_graph=True, allow_unused=True)[0]
            V_S_tau = torch.autograd.grad(V_S, tau,
                                          grad_outputs=torch.ones_like(V_S),
                                          create_graph=True, retain_graph=True, allow_unused=True)[0]
            charm = -V_S_tau
            return charm
        elif greeks == 'speed':
            # speed = dGamma/dS = d^3V/dS^2
            V_S = torch.autograd.grad(V, S,
                                      grad_outputs=torch.ones_like(V),
                                      create_graph=True, retain_graph=True, allow_unused=True)[0]
            V_SS = torch.autograd.grad(V_S, S,
                                       grad_outputs=torch.ones_like(V_S),
                                       create_graph=True, retain_graph=True, allow_unused=True)[0]
            V_SSS = torch.autograd.grad(V_SS, S,
                                        grad_outputs=torch.ones_like(V_SS),
                                        create_graph=True, retain_graph=True, allow_unused=True)[0]
            speed = V_SSS
            return speed
        elif greeks == 'color' or greeks == 'gamma_decay':
            # color = dGamma/dtau = d^3V/dS^2dtau
            V_S = torch.autograd.grad(V, S,
                                      grad_outputs=torch.ones_like(V),
                                      create_graph=True, retain_graph=True, allow_unused=True)[0]
            V_SS = torch.autograd.grad(V_S, S,
                                       grad_outputs=torch.ones_like(V_S),
                                       create_graph=True, retain_graph=True, allow_unused=True)[0]
            V_SS_tau = torch.autograd.grad(V_SS, tau,
                                           grad_outputs=torch.ones_like(V_SS),
                                           create_graph=True, retain_graph=True, allow_unused=True)[0]
            color = V_SS_tau
            return color
