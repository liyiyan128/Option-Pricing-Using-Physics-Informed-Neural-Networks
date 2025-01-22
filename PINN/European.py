import torch
# import numpy as np
from matplotlib import pyplot as plt


class EuropeanPINN(torch.nn.Module):
    def __init__(self, nn, K, T, r, sigma, S_inf=1e8, type='put', device=None):
        super(EuropeanPINN, self).__init__()
        self.loss_history = {
            'ib': [],
            'pde': [],
            'data': [],
            'total': [],
        }
        self.type = type
        self.device = device or 'cpu'
        self.V_nn = nn.to(self.device)
        self.register_buffer('K', torch.tensor(K))
        self.register_buffer('T', torch.tensor(T))
        self.register_buffer('r', torch.tensor(r))
        self.register_buffer('sigma', torch.tensor(sigma))
        self.register_buffer('S_inf', torch.tensor(S_inf))
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
        return torch.mean((self.forward(S, tau) - V)**2)

    def loss_pde(self, S, tau):
        return torch.mean(self.pde_nn(S, tau)**2)

    def loss_data(self, S, tau, V):
        if S is None:
            return torch.tensor(0.)
        return torch.mean((self.forward(S, tau) - V)**2)

    def loss(self, S_ib, tau_ib, V_ib, S_pde, tau_pde, S_data=None, tau_data=None, V_data=None):
        # return self.loss_ib(S_ib, tau_ib, V_ib) + \
        #        self.loss_pde(S_pde, tau_pde) + \
        #        self.loss_data(S_data, tau_data, V_data)
        return self.loss_ib(S_ib, tau_ib, V_ib), self.loss_pde(S_pde, tau_pde), self.loss_data(S_data, tau_data, V_data)

    def predict(self, x):
        return self.forward(x)

    def train(self, S_ib, tau_ib, V_ib, S_pde, tau_pde, S_data=None, tau_data=None, V_data=None,
              epochs=1000, optimizer='adam', verbose=True, **kwargs):
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), **kwargs)
        elif optimizer == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(self.parameters(), **kwargs)

        S_ib, tau_ib, V_ib, S_pde, tau_pde, S_data, tau_data, V_data = \
            S_ib.to(self.device), tau_ib.to(self.device), V_ib.to(self.device), \
            S_pde.to(self.device), tau_pde.to(self.device), \
            S_data.to(self.device) if S_data is not None else None, \
            tau_data.to(self.device) if tau_data is not None else None, \
            V_data.to(self.device) if V_data is not None else None

        def closure():
            self.optimizer.zero_grad()
            loss_ib, loss_pde, loss_data = self.loss(S_ib, tau_ib, V_ib, S_pde, tau_pde, S_data, tau_data, V_data)
            loss = loss_ib + loss_pde + loss_data
            loss.backward()
            return loss

        for i in range(epochs):
            self.optimizer.step(closure)

            loss_ib, loss_pde, loss_data = self.loss(S_ib, tau_ib, V_ib, S_pde, tau_pde, S_data, tau_data, V_data)
            total = loss_ib.item() + loss_pde.item()
            self.loss_history['ib'].append(loss_ib.item())
            self.loss_history['pde'].append(loss_pde.item())
            if S_data is not None:
                total += loss_data.item()
                self.loss_history['data'].append(loss_data.item())
            self.loss_history['total'].append(total)

            if verbose and (i+1) % 100 == 0:
                print(f'Epoch {i+1}/{epochs}:\nIB Loss: {loss_ib}\nPDE Loss: {loss_pde}\nData Loss: {loss_data}\n')

    def plot_loss(self, ib=True, pde=True, data=True,
                  range=-1, log_scale=True, title='Loss History', save=False, file_name='loss_history.pdf', figsize=(8, 6), fontsize=14):
        plt.figure(figsize=figsize)
        plt.plot(self.loss_history['total'][:range], label='Total loss', c='red')
        if ib:
            plt.plot(self.loss_history['ib'][:range], label='IB loss ($MSE_B$)', ls='--', alpha=0.8)
        if pde:
            plt.plot(self.loss_history['pde'][:range], label='PDE loss ($MSE_F$)', ls='--', alpha=0.8)
        if data:
            plt.plot(self.loss_history['data'][:range], label='Data loss ($MSE_{data}$)', ls='--', alpha=0.8)
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Loss', fontsize=fontsize)
        if log_scale:
            plt.yscale('log')
        plt.title(title, fontsize=fontsize+2)
        plt.legend(fontsize=fontsize)
        if save:
            plt.savefig(file_name)
        plt.show()
