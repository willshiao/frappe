from dset import save_pickle
import h5py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tqdm import trange
import torch
import time
import signal
import sys
from self_frappe import self_supervised_mirror

N_SAMPLES = 500
start_at = 0

preds = np.zeros(N_SAMPLES)
times = np.zeros(N_SAMPLES)

def sigint_handler(signal, frame):
    print('Interrupted, saving...')
    save_pickle([preds, times], '../data/500_eval_interrupted_ss.pkl')
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

try:
    with h5py.File(f'../data/diverse-ten_{N_SAMPLES}.h5', 'r', libver='latest') as f:
        ranks = np.array(f['ranks'])
        n_ten = np.array(f['n_tens'])
        for i in trange(start_at, n_ten):
            ten = torch.tensor(np.array(f[str(i)]))
            st_time = time.perf_counter()
            syn_max_noise = 0.55
            syn_min_noise = 0.02
            synth_per_rank = 10
            (feats, ranks), ten_pred = self_supervised_mirror(ten, max_rank=50, loading_bar=True, syn_min_noise=syn_min_noise, syn_max_noise=syn_max_noise, synth_per_rank=synth_per_rank)
            end_time = time.perf_counter()

            preds[i] = ten_pred
            times[i] = end_time - st_time

            print(f'MAE: @ iter {i}', mean_absolute_error(ranks[:i], preds[:i]))
            print(f'MAPE @ iter {i}: ', mean_absolute_percentage_error(ranks[:i], preds[:i]))

    preds = np.array(preds)
    times = np.array(times)

    print(f'Mean Runtime: {np.mean(times)} +/- {np.std(times)}')
    print('MAE:', mean_absolute_error(ranks, preds))
    print('MAPE: ', mean_absolute_percentage_error(ranks, preds))

    save_pickle([preds, times], '../data/500_eval_ss.pkl')
except Exception as err:
    save_pickle([preds, times], '../data/500_eval_err_ss.pkl')
    raise err
