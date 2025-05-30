import warnings
import argparse
import numpy as np
import torch
from time import time
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from PINN.VanillaOptions import VanillaOptionPINN
from PINN.utilities import V_BS, V_BS_CN, european_option_greeks, xavier_init

# --------
# OPTION PARAMETERS
TYPE = 'put'
STYLE = 'american'
K = 100
sigma = 0.3
r = 0.05
T = 1

# TRAINING PARAMETERS
S_inf = 3 * K
SCALE = K/10
N_pde = 2000
SAMPLING = 'sobol'
RESAMPLE = 0
PDE_BATCH = -1
IB_BATCH = -1
SHUFFLE = True
EPOCHS_ADAM = 500
EPOCHS_LBFGS = 500
LR_ADAM = 0.005
LR_LBFGS = 1
EARLY_STOPPING = 10

IB_DATA_PATH = f'./data/{STYLE}_{TYPE}_ib_{SAMPLING}.pt'
OUTPUT_PATH = './data/output/'
#### MODIFY ####
ARCHITECTURE = 'test'
MODEL_PATH = f'./models/{STYLE}_{TYPE}_{ARCHITECTURE}/'
FILE_NAME = f'{STYLE}_{TYPE}_{ARCHITECTURE}.npy'
# --------

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Run PINN training with parallelisation.')
parser.add_argument('--n_runs', type=int, default=10, help='Number of runs')
parser.add_argument('--n_jobs', type=int, default=5, help='Number of jobs for parallelisation')
parser.add_argument('--save_model', type=bool, default=True, help='Save model flag')
args = parser.parse_args()
N_RUNS = args.n_runs
N_JOBS = args.n_jobs
SAVE_MODEL = args.save_model

SEEDS = np.random.randint(0, 1000, N_RUNS)

greeks = ['Delta', 'Theta', 'Gamma']
results = {
    'RMSE': np.zeros(N_RUNS),
    'MPE': np.zeros(N_RUNS),
    'GREEKS': greeks,
    'TIME': np.zeros(N_RUNS),
    'SEEDS': SEEDS
}
results.update({'RMSE_' + greek: np.zeros(N_RUNS) for greek in greeks})
results.update({'MPE_' + greek: np.zeros(N_RUNS) for greek in greeks})

ib = torch.load(IB_DATA_PATH)
S_ib, tau_ib, V_ib = ib['S'], ib['tau'], ib['V']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)


def train(i, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # --------
    # PINN ARCHITECTURE
    nn = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 1)
    )
    nn.apply(xavier_init)
    # --------
    model = VanillaOptionPINN(nn, K, T, sigma, r,
                              S_inf, scale=SCALE,
                              type=TYPE, style=STYLE,
                              device=device).to(device)
    loss_weights = (1., 1., 1.)

    t = time()
    model.fit(tau_ib, S_ib, V_ib,
              tau_pde=None, S_pde=None,
              valid=False,
              N_pde=N_pde, sampling=SAMPLING, resample=RESAMPLE,
              pde_batch_size=PDE_BATCH, ib_batch_size=IB_BATCH, shuffle=SHUFFLE,
              loss_weights=loss_weights,
              epochs=EPOCHS_ADAM, early_stopping=EARLY_STOPPING, verbose=False,
              optimizer='adam', lr=LR_ADAM,)
    model.fit(tau_ib, S_ib, V_ib,
              tau_pde=None, S_pde=None,
              valid=False,
              N_pde=N_pde, sampling=SAMPLING, resample=RESAMPLE,
              pde_batch_size=PDE_BATCH, ib_batch_size=IB_BATCH, shuffle=SHUFFLE,
              loss_weights=loss_weights,
              epochs=EPOCHS_LBFGS, early_stopping=EARLY_STOPPING, verbose=False,
              optimizer='lbfgs', lr=LR_LBFGS, line_search_fn='strong_wolfe')
    t = time() - t
    if SAVE_MODEL:
        model_name = f'{STYLE}_{TYPE}_{i}.pt'
        torch.save(model, MODEL_PATH + model_name)

    S_eval = torch.linspace(0, S_inf, 1000)
    tau_eval = torch.linspace(0, T, 1000)
    S_eval, tau_eval = torch.meshgrid(S_eval, tau_eval, indexing='xy')
    S_eval.requires_grad = True
    tau_eval.requires_grad = True
    V_pred = model(tau_eval.reshape(-1, 1), S_eval.reshape(-1, 1)).detach().cpu().numpy().reshape(1000, 1000)

    S_eval_np, tau_eval_np = np.meshgrid(np.linspace(0, S_inf, 1000), np.linspace(0, T, 1000))
    if STYLE == 'european':
        V_true = V_BS(tau_eval_np, S_eval_np, K, sigma, r, TYPE)
    else:
        V_CN, t_CN, S_CN = V_BS_CN(100, 100, K, T, sigma, r, S_inf, TYPE, STYLE)
        interpolator = RegularGridInterpolator((t_CN, S_CN), V_CN, method='linear', bounds_error=False, fill_value=np.nan)
        V_true = interpolator((tau_eval_np, S_eval_np))
    V_err = V_pred - V_true
    RMSE = np.sqrt(np.mean(V_err**2))
    MPE = np.max(np.abs(V_err))

    RMSE_greeks = {}
    MPE_greeks = {}
    if STYLE == 'european':
        for greek in greeks:
            greek_pred = model.evaluate_greeks(tau_eval.reshape(-1, 1), S_eval.reshape(-1, 1), greek).detach().cpu().numpy().reshape(1000, 1000)
            greek_true = european_option_greeks(tau_eval_np, S_eval_np, K, sigma, r, greek, TYPE)
            greek_err = greek_pred - greek_true
            RMSE_greeks[greek] = np.sqrt(np.mean(greek_err**2))
            MPE_greeks[greek] = np.max(np.abs(greek_err))

    print(f'Training Run {i}    |    RMSE: {RMSE}')
    return RMSE, MPE, RMSE_greeks, MPE_greeks, t


parallel_results = Parallel(n_jobs=N_JOBS)(delayed(train)(i, SEEDS[i]) for i in tqdm(range(N_RUNS), desc='Training Progress'))

for i, (rmse, mpe, rmse_greeks, mpe_greeks, t) in enumerate(parallel_results):
    results['RMSE'][i] = rmse
    results['MPE'][i] = mpe
    if STYLE == 'european':
        for greek in greeks:
            results['RMSE_' + greek][i] = rmse_greeks[greek]
            results['MPE_' + greek][i] = mpe_greeks[greek]
    results['TIME'][i] = t

if SAVE_MODEL:
    np.save(OUTPUT_PATH + FILE_NAME, results)
    model = torch.load(MODEL_PATH + f'{STYLE}_{TYPE}_0.pt')
    print(model)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
print(ARCHITECTURE)
RMSE = results['RMSE']
idx = [i for i in range(len(RMSE)) if RMSE[i] < 10]
RMSE = RMSE[idx]
RMSE_mean = RMSE.mean()
RMSE_std = RMSE.std()
print(f'RMSE mean: {RMSE_mean:.2e}\nRMSE std: {RMSE_std:.2e}\nRMSE min: {RMSE.min():.2e}')
# MPE = np.array([mpe if mpe < 1 else 0 for mpe in results['MPE']])
# MPE_mean = MPE.mean()
# MPE_std = MPE.std()
# print(f'MPE mean: {MPE_mean:.2e}\nMPE std: {MPE_std:.2e}\nMPE min: {MPE.min():.2e}')
print(f'Average training time: {results["TIME"].mean():.2f} seconds')
