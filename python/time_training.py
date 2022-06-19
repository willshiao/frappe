import numpy as np
from lightgbm import LGBMRegressor
from dset import generate_diverse_tens
from extract_features import extract_features
import torch
import time
from tqdm import trange

tens, y = generate_diverse_tens(n=10000, min_dim=2, max_dim=100, min_rank=2, max_rank=50, min_noise=0.04, max_noise=0.20, return_tens=True)

times = []
for i in trange(10):
    start = time.perf_counter()
    feats = []
    for ten in tens:
        feats.append(extract_features(ten))

    X = torch.vstack(feats)
    model = LGBMRegressor(n_estimators=1000)
    model.fit(X, y)
    end = time.perf_counter()
    times.append(end - start)

print('Training Time: ', np.mean(times), '+/-', np.std(times))
