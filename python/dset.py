import pickle
import random

import h5py
import numpy as np
import tensorly as tl
import torch
from torch.linalg import norm
from tqdm import trange

from extract_features import extract_features
from extract_features4 import extract_features4d


def create_cp(dims, rank, sparsity=None, method='rand', weights=False, return_tensor=False, noise=None, sparse_noise=True):
    tl.set_backend('pytorch')

    if method == 'rand':
        randfunc = torch.rand
    elif method == 'randn':
        randfunc = torch.randn
    else:
        raise NotImplementedError(f'Unknown random method: {method}')

    n_dims = len(dims)
    factors = [randfunc((dim, rank)) for dim in dims]

    if sparsity is not None:
        if isinstance(sparsity, float):
            sparsity = [sparsity for _ in range(n_dims)]
        elif not isinstance(sparsity, list) and not isinstance(sparsity, tuple):
            raise ValueError('Sparsity parameter should either be a float or tuple/list.')

        # Sparsify factors
        for dim in range(n_dims):
            n_el = dims[dim] * rank
            to_del = round(sparsity[dim] * n_el)
            if to_del == 0:
                continue
            idxs = torch.tensor(random.sample(range(n_el), to_del))
            factors[dim].view(-1)[idxs] = 0
            # torch.randperm(n_el, device=device)[:n_select]

    ten = None
    # Add noise
    if noise is not None:
        ten = tl.cp_to_tensor((torch.ones(rank), factors))
        if (sparsity is None or not sparse_noise):
            nten = torch.randn(ten.size())
            ten += noise * (norm(ten) / norm(nten)) * nten
        else:
            flat = ten.view(-1)
            nzs = torch.nonzero(flat, as_tuple=True)[0]
            nvec = torch.randn(nzs.size(0))
            flat[nzs] += noise * (norm(ten) / norm(nvec)) * nvec

    if return_tensor:
        if ten is None:
            return tl.cp_to_tensor((torch.ones(rank), factors))
        return ten
    if weights:
        return torch.ones(rank), factors
    return factors

def load_dset(filename, start_at = None, max_num = None, **kwargs):
    feat_rows = []

    with h5py.File(filename, 'r') as data_f:
        ranks = torch.tensor(np.array(data_f['rank_data']))
        iter_dim = np.argmax(data_f['ten_data'].shape)
        n_tens = data_f['ten_data'].shape[iter_dim]

        if start_at is None:
            start_idx = 0
        else:
            start_idx = start_at

        if max_num is not None:
            n_tens = min(n_tens, start_idx + max_num)
            ranks = ranks.ravel()[start_idx:n_tens]

        for i in trange(start_idx, n_tens):
            if iter_dim == 0:
                data = data_f[data_f['ten_data'][i, 0]]
            else:
                data = data_f[data_f['ten_data'][0, i]]
            ten = torch.tensor(np.array(data['data']))
            sz = tuple(data['size'])
            ten = ten.permute((2, 1, 0))

            assert(sz == tuple(ten.size()))
            feat_rows.append(extract_features(ten, **kwargs))
    
    feat_mat = torch.vstack(feat_rows)
    return feat_mat.numpy(), ranks.numpy()


def load_tens(filename, start_at = None, max_num = None):
    out = []

    with h5py.File(filename, 'r') as data_f:
        ranks = torch.tensor(np.array(data_f['rank_data']))
        iter_dim = np.argmax(data_f['ten_data'].shape)
        n_tens = data_f['ten_data'].shape[iter_dim]

        if start_at is None:
            start_idx = 0
        else:
            start_idx = start_at

        if max_num is not None:
            n_tens = min(n_tens, start_idx + max_num)
            ranks = ranks.ravel()[start_idx:n_tens]

        for i in trange(start_idx, n_tens):
            if iter_dim == 0:
                data = data_f[data_f['ten_data'][i, 0]]
            else:
                data = data_f[data_f['ten_data'][0, i]]
            ten = torch.tensor(np.array(data['data']))
            sz = tuple(data['size'])
            ten = ten.permute((2, 1, 0))

            assert(sz == tuple(ten.size()))
            out.append(ten)

    return out, ranks

def generate_diverse_tens(n=10000, min_dim=1, max_dim=100, min_rank=1, max_rank=50, loading_bar=True, order=3, min_noise=0.02, max_noise=0.10, per_all_dense=0.3, per_sparse=0.2, min_sparsity=0.01, max_sparsity=0.5, return_tens=False, use_recursive_feat=True):
    all_feats = []
    all_ranks = []
    all_tens = []

    n_dense = round(per_all_dense * n)
    n_sparse = round(per_sparse * n)
    n_mixed = n - (n_dense + n_sparse)
    assert(n_mixed >= 0)

    sparse_range = trange(n_sparse, desc='Fully Sparse Samples:') if loading_bar else range(n_sparse)
    for _ in sparse_range:
        dims = [random.randint(min_dim, max_dim) for _ in range(order)]
        rank = random.randint(min_rank, max_rank)
        noise = random.uniform(min_noise, max_noise)
        sparsity = random.uniform(min_sparsity, max_sparsity)
        # try:
        ten = create_cp(dims, rank, sparsity=sparsity, noise=noise, return_tensor=True)
        if return_tens:
            all_tens.append(ten)
        else:
            if order == 3:
                all_feats.append(extract_features(ten))
            else:
                all_feats.append(extract_features4d(ten, use_recursive_feat=use_recursive_feat))
        all_ranks.append(rank)

    dense_range = trange(n_dense, desc='Dense Samples:') if loading_bar else range(n_dense)
    for _ in dense_range:
        dims = [random.randint(min_dim, max_dim) for _ in range(order)]
        rank = random.randint(min_rank, max_rank)
        noise = random.uniform(min_noise, max_noise)
        try:
            ten = create_cp(dims, rank, noise=noise, return_tensor=True)
            if return_tens:
                all_tens.append(ten)
            else:
                if order == 3:
                    all_feats.append(extract_features(ten))
                else:
                    all_feats.append(extract_features4d(ten, use_recursive_feat=use_recursive_feat))
            all_ranks.append(rank)
        except Exception as err:
            print(f'ERROR: {err}')

    mixed_range = trange(n_mixed, desc='Mixed Samples:') if loading_bar else range(n_mixed)
    for _ in mixed_range:
        dims = [random.randint(min_dim, max_dim) for _ in range(order)]
        rank = random.randint(min_rank, max_rank)
        noise = random.uniform(min_noise, max_noise)
        sparsity = random.uniform(min_sparsity, max_sparsity)
        try:
            ten = create_cp(dims, rank, sparsity=sparsity, noise=noise, sparse_noise=False, return_tensor=True)
            if return_tens:
                all_tens.append(ten)
            else:
                if order == 3:
                    all_feats.append(extract_features(ten))
                else:
                    all_feats.append(extract_features4d(ten, use_recursive_feat=use_recursive_feat))
            all_ranks.append(rank)
        except Exception as err:
            print(f'ERROR: {err}')

    if return_tens:
        return all_tens, np.array(all_ranks)

    feat_mat = torch.vstack(all_feats)
    return feat_mat.numpy(), np.array(all_ranks)


def save_pickle(obj, loc, protocol=pickle.HIGHEST_PROTOCOL):
    """Saves a pickled version of `obj` to `loc`.
    Allows for quick 1-liner to save a pickle without leaving a hanging file handle.
    Useful for Jupyter notebooks.

    Also behaves differently to pickle in that it defaults to pickle.HIGHEST_PROTOCOL
        instead of pickle.DEFAULT_PROTOCOL.

    Arguments:
        obj {Any} -- The object to be pickled.
        loc {Path|str} -- A location to save the object to.
        protocol {pickle.Protocol} -- The pickle protocol level to use.
            (default {pickle.HIGHEST_PROTOCOL})
    """
    with open(loc, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)

def load_pickle(loc):
    """Loads a pickled object from `loc`.
    Very helpful in avoiding overwritting a pickle when you read using 'wb'
    instead of 'rb' at 3 AM.
    Also provides a convinient 1-liner to read a pickle without leaving an open file handle.
    If we encounter a PickleError, it will try to use pickle5.

    Arguments:
        loc {Path|str} -- A location to read the pickled object from.

    Returns:
        Any -- The pickled object.
    """
    try:
        with open(loc, 'rb') as f:
            return pickle.load(f)
    except pickle.PickleError:
        # Maybe it's a pickle5 and we use Python <= 3.8.3
        import pickle5
        with open(loc, 'rb') as f:
            return pickle5.load(f)
