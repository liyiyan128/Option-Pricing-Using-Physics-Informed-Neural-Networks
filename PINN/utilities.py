import torch
import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import QuantLib as ql


def V_BS(tau, S, K, r, sigma, type='put'):
    type = type.lower()

    epsilon = max(np.finfo(float).eps, 1e-16)
    tau = np.where(tau < epsilon, epsilon, tau)
    S = np.where(S < epsilon, epsilon, S)

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    d2 = (np.log(S/K) + (r - 0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))

    if type == 'put':
        res = K*np.exp(-r*tau)*norm.cdf(-d2) - S*norm.cdf(-d1)
    elif type == 'call':
        res = S*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)

    return res


def V_BS_CN(m, n,  K, T,  r, sigma, S_inf,
            type='put', style='european'):
    """
    Cranck-Nicolson V(t, S)
    """
    type = type.lower()
    style = style.lower()

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
        V[:, -1] = S_inf - K * np.exp(-r * t)

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
        if style == 'american':
            if type == 'put':
                intrinsic = np.maximum(K - S[1:-1], 0)
            elif type == 'call':
                intrinsic = np.maximum(S[1:-1] - K, 0)
            V_next = np.maximum(V_next, intrinsic)
        V[i+1, 1:-1] = V_next

    return V, S, t


def V_quantlib(tau, S, K, T, r, sigma, q=0,
               type='put', style='american',
               model='black_scholes', method=None, **kwargs):
    t = T - tau
    t, S = np.meshgrid(t, S)
    shape = t.shape
    t = t.flatten()
    S = S.flatten()
    u = ql.SimpleQuote(S[0])
    r = ql.SimpleQuote(r)
    sigma = ql.SimpleQuote(sigma)
    type = type.lower()
    style = style.lower()
    model = model.lower()
    if method is None:
        method = 'bs' if style == 'european' else 'crr'
    else:
        method = method.lower()

    calendar = kwargs.get('calendar', ql.NullCalendar())
    dayCounter = kwargs.get('dayCounter', ql.Actual365Fixed())
    today = ql.Date(1, 1, 2025)  # ql.Date.todaysDate()
    expiry = today + int(T * 365)
    ql.Settings.instance().evaluationDate = today + int(t[0] * 365)

    option_type = ql.Option.Put if type == 'put' else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type, K)
    exercise = ql.EuropeanExercise(expiry) if style == 'european' else \
               ql.AmericanExercise(today, expiry)
    option = ql.VanillaOption(payoff, exercise)

    spotHandle = ql.QuoteHandle(u)
    if model in ['black_scholes', 'bs']:
        riskFreeCurve = ql.FlatForward(0, calendar, ql.QuoteHandle(r), dayCounter)
        riskFreeTS = ql.YieldTermStructureHandle(riskFreeCurve)
        volCurve = ql.BlackConstantVol(0, calendar, ql.QuoteHandle(sigma), dayCounter)
        volTS = ql.BlackVolTermStructureHandle(volCurve)
        if q == 0:
            process = ql.BlackScholesProcess(spotHandle, riskFreeTS, volTS)
        else:
            dividendCurve = ql.FlatForward(0, calendar, ql.QuoteHandle(r), dayCounter)
            dividendTS = ql.YieldTermStructureHandle(dividendCurve)
            process = ql.BlackScholesMertonProcess(spotHandle, dividendTS, riskFreeTS, volTS)
            
    if method in ['bs', 'black_scholes', 'analytic']:
        engine = ql.AnalyticEuropeanEngine(process)
    elif method in ['crr', 'binomial', 'tree', 'binomial_tree']:
        steps = kwargs.get('steps', 200)
        engine = ql.BinomialVanillaEngine(process, 'crr', steps)
    elif method in ['mc', 'monte_carlo']:
        timeSteps = kwargs.get('timeSteps', 20)
        requiredSamples = kwargs.get('requiredSamples', 250000)
        if style == 'american':
            engine = ql.MCAmericanEngine(process, #'PseudoRandom', 
                                         timeSteps=timeSteps,
                                         requiredSamples=requiredSamples)
        elif style == 'european':
            engine = ql.MCEuropeanEngine(process, #"PseudoRandom",
                                         timeSteps=timeSteps,
                                         requiredSamples=requiredSamples)
    
    option.setPricingEngine(engine)
    V = np.zeros_like(t)
    for i in range(len(t)):
        ql.Settings.instance().evaluationDate = today + int(t[i] * 365)
        u.setValue(S[i])
        V[i] = option.NPV()
    return V.reshape(shape), (T-t).reshape(shape), S.reshape(shape)


def european_option_greeks(tau, S, K, r, sigma, greeks='delta', type='put'):
    greeks = greeks.lower()
    type = type.lower()

    epsilon = max(np.finfo(float).eps, 1e-16)
    tau = np.where(tau < epsilon, epsilon, tau)
    S = np.where(S < epsilon, epsilon, S)

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    d2 = (np.log(S/K) + (r - 0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))

    if greeks == 'delta':
        if type == 'put':
            delta = -norm.cdf(-d1)
        elif type == 'call':
            delta = norm.cdf(d1)
        res = delta
    elif greeks == 'vega':
        vega = K*np.exp(-r*tau)*norm.pdf(d2)*np.sqrt(tau)
        res = vega
    elif greeks == 'theta' or greeks == 'time_decay':
        if type == 'put':
            theta = -S*norm.pdf(d1)*sigma/(2*np.sqrt(tau)) + r*K*np.exp(-r*tau)*norm.cdf(-d2)
        elif type == 'call':
            theta = -S*norm.pdf(d1)*sigma/(2*np.sqrt(tau)) - r*K*np.exp(-r*tau)*norm.cdf(d2)
        res = theta
    elif greeks == 'rho':
        if type == 'put':
            rho = -K*tau*np.exp(-r*tau)*norm.cdf(-d2)
        elif type == 'call':
            rho = K*tau*np.exp(-r*tau)*norm.cdf(d2)
        res = rho
    elif greeks == 'gamma':
        gamma = norm.pdf(d1)/(S*sigma*np.sqrt(tau))
        res = gamma
    elif greeks == 'charm' or greeks == 'delta_decay':
        charm = -norm.pdf(d1)*((2*r*np.sqrt(tau) - d2*sigma*np.sqrt(tau))/(2*tau*sigma*np.sqrt(tau)))
        res = charm
    elif greeks == 'speed':
        speed = -norm.pdf(d1)/(S**2*sigma*np.sqrt(tau))*(d1/(sigma*np.sqrt(tau)) + 1)
        res = speed
    elif greeks == 'color' or greeks == 'gamma_decay':
        color = -norm.pdf(d1)/(2*S*tau*sigma*np.sqrt(tau))*(1 + d1*(2*r*tau-d2*sigma*np.sqrt(tau))/(sigma*np.sqrt(tau)))
        res = color

    if np.isnan(res).any():
        # print corresponding tau and S of fisrt nan
        idx = np.where(np.isnan(res))[0][0]
        print(f'NaN in {greeks} at tau={tau[idx]}, S={S[idx]}')
        raise ValueError(f'NaN in {greeks}')
    return res


def collocation_points(model, N_pde=2000,
                       sampling='sobol', sobol=None,
                       grid=(1000, 1000), alpha_beta_tau=(.5, .5),
                       adaptive_base='sobol', tau_pde_base=None, S_pde_base=None,
                       **kwargs):
    """
    model: PINN model
    sobol: sobol engine
    grid: grid size (tau, S) for pde error evaluation for adaptive sampling
    alpha_beta_tau: alpha and beta for beta distribution for tau in adaptive sampling
    adaptive_base: base sampling for adaptive sampling
    tau_pde_base: base samples for tau in adaptive sampling
    S_pde_base: base samples for S in adaptive sampling
    """
    grid = kwargs.get('grid', grid)
    alpha_beta_tau = kwargs.get('alpha_beta_tau', alpha_beta_tau)
    adaptive_base = kwargs.get('adaptive_base', adaptive_base)

    S_pde = None
    tau_pde = None

    if sampling == 'uniform':
        S_pde = torch.rand(N_pde, 1)*model.S_inf
        tau_pde = torch.rand(N_pde, 1)*model.T
    elif sampling == 'sobol':
        if sobol is None:
            raise ValueError('sobol engine is required')
        sobol_samples = sobol.draw(N_pde)
        S_pde = sobol_samples[:, 0].reshape(-1, 1)*model.S_inf
        tau_pde = sobol_samples[:, 1].reshape(-1, 1)*model.T
    elif sampling == 'importance':
        n_samples = N_pde // 2
        n1 = n_samples // 2
        n2 = n_samples - n1
        # normal for S centred at K
        S_pde = torch.normal(mean=model.K, std=model.K/4, size=(n1, 1))
        # exp dist for S biased towards 0
        S_pde = torch.concat([S_pde, torch.distributions.exponential.Exponential(2).sample((n2, 1))], dim=0)
        S_pde = torch.clamp(S_pde, 0, model.S_inf)  # S in [0, S_inf]
        # beta dist for tau biased towards 0
        # try (alpha, beta) = (.5, .5), (2, 5)
        tau_pde = torch.distributions.beta.Beta(alpha_beta_tau[0], alpha_beta_tau[1]).sample((n_samples, 1))*model.T
        # # exponential dist for tau
        # tau_pde = torch.distributions.exponential.Exponential(1/(T/4)).sample((n_samples, 1))*T
        tau_pde = torch.clamp(tau_pde, 0, model.T)  # tau in [0, T]

        # base samples
        if sobol is not None:
            sobol = torch.quasirandom.SobolEngine(dimension=2)
            base_samples = sobol.draw(N_pde - n_samples)
        else:
            base_samples = torch.concat([torch.rand(N_pde - n_samples, 1), torch.rand(N_pde - n_samples, 1)], dim=1)
        # mix with base_samples
        S_pde = torch.cat([S_pde, base_samples[:, 0].reshape(-1, 1)*model.S_inf], dim=0)
        tau_pde = torch.cat([tau_pde, base_samples[:, 1].reshape(-1, 1)*model.T], dim=0)
    elif sampling == 'adaptive':
        if S_pde_base is None or tau_pde_base is None:
            # sobol/uniform as base samples for adaptive sampling
            print(f'Use {adaptive_base} as base samples for adaptive sampling')
            tau_pde, S_pde = collocation_points(model, N_pde, sampling=adaptive_base, sobol=sobol, **kwargs)
        else:
            S_eval = torch.linspace(0, model.S_inf, grid[1])
            tau_eval = torch.linspace(0, model.T, grid[0])
            S_eval, tau_eval = torch.meshgrid(S_eval, tau_eval, indexing='xy')
            S_eval.requires_grad = True
            tau_eval.requires_grad = True
            pde_err_abs = model.pde_nn(S_eval.reshape(-1, 1), tau_eval.reshape(-1, 1)).reshape(grid).abs().detach().numpy()
            # Scott's rule n**(-1./(d+4))
            sigma_scott = pde_err_abs.std() * pde_err_abs.size ** (-1/6)
            # Apply Gaussian filter
            pde_err_abs = gaussian_filter(pde_err_abs, sigma=sigma_scott)
            pde_err_abs = np.where(np.isnan(pde_err_abs), 0, pde_err_abs)  # remove nan

            # sample from pde_err_abs
            n_samples = int(0.75*N_pde)
            probs = pde_err_abs.flatten() / pde_err_abs.flatten().sum()
            probs = np.where(np.isnan(probs), 0, probs)  # remove nan
            if np.all(probs == 0):
                print(f'Zero probabilities, resample using {sampling}')
                tau_pde, S_pde = collocation_points(model, N_pde, sampling=adaptive_base, sobol=sobol, **kwargs)
            else:
                idx = np.random.choice(grid[0]*grid[1], size=n_samples, p=probs)
                S_pde_resample = S_eval.flatten()[idx].reshape(-1, 1)
                tau_pde_resample = tau_eval.flatten()[idx].reshape(-1, 1)

                # Maintain original data size
                idx = np.random.choice(N_pde, size=N_pde - n_samples, replace=False)
                S_pde = torch.cat([S_pde_resample, S_pde_base[idx]])
                tau_pde = torch.cat([tau_pde_resample, tau_pde_base[idx]])

    S_pde = S_pde.detach().requires_grad_(True)
    tau_pde = tau_pde.detach().requires_grad_(True)
    return tau_pde, S_pde


if __name__ == '__main__':
    # parameters
    K = 4
    sigma = 0.3
    r = 0.03
    T = 1
    S_inf = 3 * K

    # test european_option_greeks numerical stability
    S_eval_np = np.linspace(0, S_inf, 1000)
    tau_eval_np = np.linspace(0, T, 1000)
    S_eval_np, tau_eval_np = np.meshgrid(S_eval_np, tau_eval_np)

    greeks = ['Delta', 'Theta', 'Gamma', 'Charm', 'Speed', 'Color']
    for greek in greeks:
        greek_true = european_option_greeks(tau_eval_np, S_eval_np, K, r, sigma, greek, 'put')
