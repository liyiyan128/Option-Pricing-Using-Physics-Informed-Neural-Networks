import numpy as np
import scipy.sparse as sp
from scipy.stats import norm


def V_BS(S, tau, K, r, sigma, type='put'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)

    if type == 'put':
        res = K*np.exp(-r*tau)*norm.cdf(-d2) - S*norm.cdf(-d1)
    elif type == 'call':
        res = S*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)

    return res


def V_BS_CN(m, n, T, K, sigma, r, S_inf, type='put'):
    """
    Cranck-Nicolson V(t, S)
    """
    S = np.linspace(0, S_inf, m+1)
    t = np.linspace(0, T, n+1)
    dt = T / n

    # Initialize the solution grid
    V = np.zeros((n+1, m+1))
    # Set initial and boundary conditions
    if type == 'put':
        V[0, :] = np.maximum(K - S, 0)  # at maturity
        V[:, 0] = K * np.exp(-r * t)  # S = 0
        V[:, -1] = 0  # S = S_inf
    elif type == 'call':
        V[0, :] = np.maximum(S - K, 0)
        V[:, 0] = 0
        V[:, -1] = S_inf

    # construct matrix
    D1 = sp.diags(np.arange(1, m)).tocsc()
    D2 = sp.diags(np.arange(1, m)**2).tocsc()
    T1 = sp.diags([-1, 0, 1], [-1, 0, 1], shape=(m-1, m-1)).tocsc()
    T2 = sp.diags([1, -2, 1], [-1, 0, 1], shape=(m-1, m-1)).tocsc()
    I = sp.eye(m-1).tocsc()
    B = (1+r*dt)*I - 0.5*dt*sigma**2*D2.dot(T2) - 0.5*dt*r*D1.dot(T1)
    F = (1-r*dt)*I + 0.5*dt*sigma**2*D2.dot(T2) + 0.5*dt*r*D1.dot(T1)
    M1 = I + B
    M2 = I + F

    # M1 V_i+1 = M2 V_i + c
    for i in range(n):
        c = np.zeros(m-1)
        c[0] = 0.5*dt*(sigma**2 - r)*(V[i+1, 0] + V[i, 0])
        c[-1] = 0.5*dt*((m-1)*sigma**2 + r)*(V[i+1, -1] + V[i, -1])

        RHS = M2.dot(V[i, 1:-1]) + c
        V_next = sp.linalg.spsolve(M1, RHS)
        V[i+1, 1:-1] = V_next

    return V, S, t
