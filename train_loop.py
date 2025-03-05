import warnings
import argparse
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from PINN.VanillaOptions import VanillaOptionPINN
from PINN.utilities import V_BS, european_option_greeks

# --------
# OPTION PARAMETERS
TYPE = 'put'
STYLE = 'european'
K = 4
sigma = 0.3
r = 0.03
T = 1
S_inf = 3 * K
N_pde = 2500
# --------

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Run PINN training with parallelisation.')
parser.add_argument('--n_runs', type=int, default=100, help='Number of runs')
parser.add_argument('--save_model', type=bool, default=True, help='Save model flag')
parser.add_argument('--model_path', type=str, default='./models/european_put/', help='Path to save models')
parser.add_argument('--ib_data_path', type=str, default='./data/european_put_ib_sobol.pt', help='Path to input boundary data')
parser.add_argument('--output_path', type=str, default='./data/output/', help='Path to save results')
parser.add_argument('--file_name', type=str, default='results.npy', help='Output file name')
parser.add_argument('--epochs_adam', type=int, default=200, help='Number of epochs for Adam optimizer')
parser.add_argument('--epochs_lbfgs', type=int, default=300, help='Number of epochs for L-BFGS optimizer')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for Adam optimizer')
args = parser.parse_args()

N_RUNS = args.n_runs
SAVE_MODEL = args.save_model
MODEL_PATH = args.model_path
IB_DATA_PATH = args.ib_data_path
OUTPUT_PATH = args.output_path
FILE_NAME = args.file_name
EPOCHS_ADAM = args.epochs_adam
EPOCHS_LBFGS = args.epochs_lbfgs
LR = args.lr

SEEDS = np.random.randint(0, 1000, N_RUNS)

greeks = ['Delta', 'Theta', 'Gamma', 'Charm', 'Speed', 'Color']
results = {
    'RMSE': np.zeros(N_RUNS),
    'max_pointwise_error': np.zeros(N_RUNS),
    'SEEDS': SEEDS
}
results.update({'RMSE_' + greek: np.zeros(N_RUNS) for greek in greeks})
results.update({'max_pointwise_error_' + greek: np.zeros(N_RUNS) for greek in greeks})

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
            torch.nn.Linear(2, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 1)
    )
    # --------
    model = VanillaOptionPINN(nn, K, T, r, sigma, S_inf,
                              type=TYPE, style=STYLE,
                              device=device).to(device)
    loss_weights = (1., 1., 1.)

    model.fit(tau_ib, S_ib, V_ib,
              tau_pde=None, S_pde=None,
              valid=False,
              N_pde=N_pde, sampling='sobol', resample=0,
              loss_weights=loss_weights,
              epochs=200, verbose=False,
              optimizer='adam', lr=0.005)
    model.fit(tau_ib, S_ib, V_ib,
              tau_pde=None, S_pde=None,
              valid=False,
              N_pde=N_pde, sampling='sobol', resample=0,
              loss_weights=loss_weights,
              epochs=300, verbose=False,
              optimizer='lbfgs', line_search_fn='strong_wolfe')

    if SAVE_MODEL:
        model_name = f'{STYLE}_{TYPE}_{i}.pt'
        torch.save(model, MODEL_PATH + model_name)

    S_eval = torch.linspace(0, S_inf, 1000)
    tau_eval = torch.linspace(0, T, 1000)
    S_eval, tau_eval = torch.meshgrid(S_eval, tau_eval, indexing='xy')
    V_pred = model(tau_eval.reshape(-1, 1), S_eval.reshape(-1, 1)).detach().cpu().numpy().reshape(1000, 1000)

    S_eval_np, tau_eval_np = np.meshgrid(np.linspace(0, S_inf, 1000), np.linspace(0, T, 1000))
    V_true = V_BS(tau_eval_np, S_eval_np, K, r, sigma, TYPE)
    V_err = V_pred - V_true
    RMSE = np.sqrt(np.mean(V_err**2))
    max_pointwise_error = np.max(np.abs(V_err))

    RMSE_greeks = {}
    max_pointwise_error_greeks = {}
    for greek in greeks:
        greek_pred = model.evaluate_greeks(tau_eval.reshape(-1, 1), S_eval.reshape(-1, 1), greek).detach().cpu().numpy().reshape(1000, 1000)
        greek_true = european_option_greeks(tau_eval_np, S_eval_np, K, r, sigma, greek, TYPE)
        greek_err = greek_pred - greek_true
        RMSE_greeks[greek] = np.sqrt(np.mean(greek_err**2))
        max_pointwise_error_greeks[greek] = np.max(np.abs(greek_err))

    print(f'Training Run {i}    |    RMSE: {RMSE}')

    return RMSE, max_pointwise_error, RMSE_greeks, max_pointwise_error_greeks

parallel_results = Parallel(n_jobs=-1)(delayed(train)(i, SEEDS[i]) for i in tqdm(range(N_RUNS), desc='Training Progress'))

for i, (rmse, mpe, rmse_greeks, mpe_greeks) in enumerate(parallel_results):
    results['RMSE'][i] = rmse
    results['max_pointwise_error'][i] = mpe
    for greek in greeks:
        results['RMSE_' + greek][i] = rmse_greeks[greek]
        results['max_pointwise_error_' + greek][i] = mpe_greeks[greek]

np.save(OUTPUT_PATH + FILE_NAME, results)