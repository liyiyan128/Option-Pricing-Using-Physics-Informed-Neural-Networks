import torch
import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import QuantLib as ql


def V_BS(tau, S, K, sigma, r, type='put'):
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


def V_BS_CN(m, n,  K, T,  sigma, r, S_inf,
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
    if type == 'call':
        V[0, :] = np.maximum(S - K, 0)
        V[:, 0] = 0
        V[:, -1] = S_inf - K * np.exp(-r * t)

    if style == 'american':
        intrinsic = np.where(type == 'put',
                             np.maximum(K - S, 0),
                             np.maximum(S - K, 0))
        V = np.maximum(V, intrinsic)

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

    return V, t, S


def V_quantlib(tau, S, K, T, sigma, r, q=0,
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
        requiredSamples = kwargs.get('requiredSamples', 10000)
        if style == 'american':
            engine = ql.MCAmericanEngine(process, 'PseudoRandom', 
                                         timeSteps=timeSteps,
                                         requiredSamples=requiredSamples)
        elif style == 'european':
            engine = ql.MCEuropeanEngine(process, "PseudoRandom",
                                         timeSteps=timeSteps,
                                         requiredSamples=requiredSamples)

    option.setPricingEngine(engine)
    V = np.zeros_like(t)
    for i in range(len(t)):
        ql.Settings.instance().evaluationDate = today + int(t[i] * 365)
        u.setValue(S[i])
        V[i] = option.NPV()
    return V.reshape(shape), (T-t).reshape(shape), S.reshape(shape)


def european_option_greeks(tau, S, K, sigma, r, greeks='delta', type='put'):
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
                       **kwargs):
    """
    model: PINN model
    sobol: sobol engine
    """
    adaptive_base = kwargs.get('adaptive_base', 0.1)

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
    elif sampling == 'adaptive':
        tau_eval, S_eval = collocation_points(model, N_pde*2, sampling='sobol', sobol=torch.quasirandom.SobolEngine(dimension=2))
        pde_err_abs = model.pde_nn(S_eval.reshape(-1, 1), tau_eval.reshape(-1, 1)).abs().detach().numpy()
        # Scott's rule n**(-1./(d+4))
        sigma_scott = pde_err_abs.std() * pde_err_abs.size ** (-1/6)
        # Apply Gaussian filter
        pde_err_abs = gaussian_filter(pde_err_abs, sigma=sigma_scott).flatten()
        pde_err_abs = np.where(np.isnan(pde_err_abs), 0, pde_err_abs)  # remove nan

        # sample from pde_err_abs
        n_samples = int((1-adaptive_base)*N_pde)
        probs = pde_err_abs / pde_err_abs.sum()
        probs = np.where(np.isnan(probs), 0, probs)  # remove nan
        if np.all(probs == 0):
            print(f'Zero probabilities, resample using {sampling}')
            tau_pde, S_pde = collocation_points(model, N_pde, sampling=adaptive_base, sobol=sobol, **kwargs)
        else:
            idx = np.random.choice(pde_err_abs.size, size=n_samples, replace=False, p=probs)
            S_pde_resample = S_eval[idx]
            tau_pde_resample = tau_eval[idx]
            # Maintain original data size
            tau_base, S_base = collocation_points(model, N_pde-n_samples, sampling='sobol', sobol=torch.quasirandom.SobolEngine(dimension=2))
            S_pde = torch.cat([S_pde_resample, S_base])
            tau_pde = torch.cat([tau_pde_resample, tau_base])

    S_pde = S_pde.detach().requires_grad_(True)
    tau_pde = tau_pde.detach().requires_grad_(True)
    return tau_pde, S_pde


def collocation_points_v2(model, N_pde=2000,
                          sampling='sobol', sobol=None,
                          **kwargs):
    """
    model: PINN model
    sobol: sobol engine
    """

    S_pde = None
    tau_pde = None
    K_pde = None
    T_pde = None

    if sampling == 'sobol':
        if sobol is None:
            raise ValueError('sobol engine is required')
        sobol_samples = sobol.draw(N_pde)
        S_pde = sobol_samples[:, 0].reshape(-1, 1)*model.S_inf
        tau_pde = sobol_samples[:, 1].reshape(-1, 1)*model.T
        if model.K_min is not None:
            K_pde = sobol_samples[:, 2].reshape(-1, 1)*(model.K_max-model.K_min) + model.K_min
        if model.T_min is not None:
            T_pde = sobol_samples[:, 3].reshape(-1, 1)*(model.T_max-model.T_min) + model.T_min
    else:
        raise NotImplementedError(f'{sampling} is not implemented')

    S_pde = S_pde.detach().requires_grad_(True)
    tau_pde = tau_pde.detach().requires_grad_(True)
    if K_pde is not None:
        K_pde = K_pde.detach().requires_grad_(True)
    if T_pde is not None:
        T_pde = T_pde.detach().requires_grad_(True)
    return tau_pde, S_pde, K_pde, T_pde


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
        greek_true = european_option_greeks(tau_eval_np, S_eval_np, K, sigma, r, greek, 'put')


def xavier_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
