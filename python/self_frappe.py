from lightgbm.sklearn import LGBMRegressor
from extract_features import extract_features
from extract_features4 import extract_features4d
import random
from dset import create_cp
import torch
import numpy as np
from tqdm import trange
import tensorly as tl
tl.set_backend('pytorch')

def subsample_factors(cp_ten, n_factors=10):
    weights, facts = cp_ten
    out = []
    n_cols = facts[0].size(1)
    w = torch.ones(n_cols)

    for fact in facts:
        idxs = torch.multinomial(w, n_factors)
        out.append(fact[:, idxs])
    return (w[:n_factors], out)

def self_supervised_mirror(ten, n_estimators=1000, max_rank=20, synth_per_rank = 10, loading_bar=False, return_model=False, syn_min_noise=0, syn_max_noise=0.1):
    model = LGBMRegressor(n_estimators=n_estimators)
    out = []
    ranks = []
    ndim = len(ten.size())
    n_el = torch.numel(ten)
    is_zero = (ten == 0).sum()
    base_sparsity = is_zero / n_el

    synth_range = trange(1, max_rank, desc='Generating synthetic samples') if loading_bar else range(1, max_rank)
    for r in synth_range:
        for n in range(synth_per_rank):
            # selection = (r * synth_per_rank + n) % 3
            sparsity = float(base_sparsity)
            noise = random.uniform(syn_min_noise, syn_max_noise)
            sparse_noise = True

            new_ten = create_cp(dims=list(ten.shape), rank=r, return_tensor=True, sparsity=sparsity, sparse_noise=sparse_noise, noise=noise)
            # new_ten[sparsity_mask] = 0
            new_feats = extract_features(new_ten)
            out.append(new_feats)
            ranks.append(r)

    feats = torch.vstack(out).numpy()
    ranks = np.array(ranks)
    model.fit(feats, ranks)

    if ndim == 3:
        ten_feats = extract_features(ten).view(1, -1).numpy()
    elif ndim == 4:
        ten_feats = extract_features4d(ten).view(1, -1).numpy()
    else:
        raise ValueError('Unknown number of dimensions')
    ten_pred = model.predict(ten_feats).ravel()[0]

    ret = (feats, ranks), ten_pred
    if return_model:
        return (*ret, model)
    return ret
