
import numpy as np
import scipy.io as sio
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm

from extract_features import extract_features
from self_frappe import self_supervised_mirror

mat = sio.loadmat('../data/known_rank.mat')
ranks, tens = mat['ranks'].ravel(), mat['tensors'].ravel()
s_idxs, s_tens, s_ranks = zip(*[(i, tens[i], ranks[i]) for i in range(len(tens)) if tens[i].size <= 1e5])

converted_tens = []
conv_ranks = []
for i in range(len(s_tens)):
    if s_tens[i].dtype == np.float64:
        print('used idx', i, 'with size:', s_tens[i].shape)
        converted_tens.append(s_tens[i])
        conv_ranks.append(s_ranks[i])
        continue
    print('used idx', i, 'with size:', s_tens[i].shape)
    converted_tens.append(s_tens[i].astype(np.float64))
    conv_ranks.append(s_ranks[i])

tens = [torch.tensor(x) for x in converted_tens]
all_feats = []
idxs = []
target_tens = []
target_ranks = []

for idx, ten in tqdm(enumerate(tens)):
    if isinstance(ten, torch.ByteTensor) or isinstance(ten, torch.IntTensor):
        print('hit!', ten.type())
        continue
    print(ten.shape)
    
    ten = ten.contiguous()
    idxs.append(idx)
    ten_feat = extract_features(ten)
    all_feats.append(ten_feat)
    target_tens.append(ten)
    target_ranks.append(conv_ranks[idx])

target_ranks = np.array(target_ranks)

syn_max_noise = 0.55
syn_min_noise = 0.02
synth_per_rank = 10

ss_res = [(self_supervised_mirror(ten.contiguous(), max_rank=target_ranks[idx]*2, loading_bar=False,
    syn_min_noise=syn_min_noise, syn_max_noise=syn_max_noise, synth_per_rank=synth_per_rank)[1]) for idx, ten in enumerate(tens)]
res = np.array(ss_res)
mae = float(mean_absolute_error(target_ranks, res))
mape = float(mean_absolute_percentage_error(target_ranks, res))

print('MAE: ', mae)
print('MAPE: ', mape)
print(res)
print(target_ranks)
