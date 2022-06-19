from self_frappe import self_supervised_mirror
import scipy.io as sio
import torch

data_path = '../data/trained_alexnet_layer-10.mat'

syn_max_noise = 0.55
syn_min_noise = 0.02
synth_per_rank = 10

data = torch.tensor(sio.loadmat(data_path)['data'])
(feats, ranks), ten_pred = self_supervised_mirror(data, max_rank=100, loading_bar=True, syn_min_noise=syn_min_noise, syn_max_noise=syn_max_noise, synth_per_rank=synth_per_rank)

print('Predicted rank:', ten_pred)
