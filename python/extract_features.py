import torch
import numpy as np

FEATURE_NAMES = [
    'dimension_I',
    'dimension_J',
    'dimension_K',
    'number_of_non-zeros',
    'min_number_of_non-zeros_over_all_(1-2)-mode_slices',
    'median_number_of_non-zeros_over_all_(1-2)-mode_slices',
    'max_number_of_non-zeros_over_all_(1-2)-mode_slices',
    'min_number_of_non-zeros_over_all_(1-3)-mode_slices',
    'median_number_of_non-zeros_over_all_(1-3)-mode_slices',
    'max_number_of_non-zeros_over_all_(1-3)-mode_slices',
    'min_number_of_non-zeros_over_all_(2-3)-mode_slices',
    'median_number_of_non-zeros_over_all_(2-3)-mode_slices',
    'max_number_of_non-zeros_over_all_(2-3)-mode_slices',
    *[f'min_rank_over_all_(1-2)-mode_slices_with_thresh={x}' for x in np.round(np.arange(0.1, 1.0, 0.1), 1)],
    *[f'min_nnz_over_(1-2)-mode_slices_with_thresh={x}' for x in np.round(np.arange(0.1, 1.0, 0.1), 1)],
    *[f'median_nnz_over_(1-2)-mode_slices_with_thresh={x}' for x in np.round(np.arange(0.1, 1.0, 0.1), 1)],
    *[f'max_nnz_over_(1-3)-mode_slices_with_thresh={x}' for x in np.round(np.arange(0.1, 1.0, 0.1), 1)],
    *[f'min_nnz_over_(1-3)-mode_slices_with_thresh={x}' for x in np.round(np.arange(0.1, 1.0, 0.1), 1)],
    *[f'median_nnz_over_(1-3)-mode_slices_with_thresh={x}' for x in np.round(np.arange(0.1, 1.0, 0.1), 1)],
    *[f'max_nnz_over_(2-3)-mode_slices_with_thresh={x}' for x in np.round(np.arange(0.1, 1.0, 0.1), 1)],
    *[f'min_nnz_over_(2-3)-mode_slices_with_thresh={x}' for x in np.round(np.arange(0.1, 1.0, 0.1), 1)],
    *[f'median_nnz_over_(2-3)-mode_slices_with_thresh={x}' for x in np.round(np.arange(0.1, 1.0, 0.1), 1)],
    'max_corr_over_(1-2)-mode_slices',
    'min_corr_over_(1-2)-mode_slices',
    'median_corr_over_(1-2)-mode_slices',
    'max_corr_over_(1-3)-mode_slices',
    'min_corr_over_(1-3)-mode_slices',
    'median_corr_over_(1-3)-mode_slices',
    'max_corr_over_(2-3)-mode_slices',
    'min_corr_over_(2-3)-mode_slices',
    'median_corr_over_(2-3)-mode_slices'
]

def three_samp(vec):
    filt_vec = vec[torch.isfinite(vec)]
    if filt_vec.numel() == 0:
        return 0, 0, 0
    return torch.min(filt_vec), torch.median(filt_vec), torch.max(filt_vec)

def three_samp_alt(vec):
    filt_vec = vec[torch.isfinite(vec)]
    if filt_vec.numel() == 0:
        return 0, 0, 0
    return torch.max(filt_vec), torch.min(filt_vec), torch.median(filt_vec)

def corr_pairs(X, axis):
    n = X.size(axis)
    corrs = []
    centered_slices = [X.select(axis, i) - torch.mean(X.select(axis, i)) for i in range(n)]
    summed = [torch.sum(torch.square(centered_slices[i])) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            num = torch.dot(centered_slices[i].view(-1), centered_slices[j].view(-1))
            denom = torch.sqrt(summed[i] * summed[j])
            corrs.append(num / denom)
    return torch.tensor(corrs)

# norm_type can either be min-max (default) or z
def extract_features(ten, norm_type='min-max', calculate_rank=True):
    feat = torch.empty(112)
    feat[:] = float('nan')

    # dimensions i, j, k
    feat[0], feat[1], feat[2] = ten.shape

    # nnz
    feat[3] = (ten >= 0.00001).sum()

    # get nnz per slice along (1,2)-mode slices
    feat[4], feat[5], feat[6] = three_samp(torch.count_nonzero(ten, (0,1)))
    # get nnz per slice along (1,3)-mode slices
    feat[7], feat[8], feat[9] = three_samp(torch.count_nonzero(ten, (0,2)))
    # get nnz per slice along (2,3)-mode slices
    feat[10], feat[11], feat[12] = three_samp(torch.count_nonzero(ten, (1,2)))

    if calculate_rank:
        # get rank along (1,2)-mode slices
        m12_ranks = torch.linalg.matrix_rank(torch.transpose(ten, 0, 2))
        feat[13], feat[14], feat[15] = three_samp(m12_ranks)

        # get rank along (1,3)-mode slices
        m13_ranks = torch.linalg.matrix_rank(torch.transpose(ten, 0, 1))
        feat[16], feat[17], feat[18] = three_samp(m13_ranks)

        # get rank along (2,3)-mode slices
        m23_ranks = torch.linalg.matrix_rank(ten)
        feat[19], feat[20], feat[21] = three_samp(m23_ranks)
    else:
        feat[13:22] = 0

    # normalize tensor
    norm_ten = torch.clone(ten).detach()
    if norm_type == 'min-max':
        min_ten = torch.min(ten)
        max_ten = torch.max(ten)
        norm_ten = (norm_ten - min_ten) / (max_ten - min_ten)
    elif norm_type == 'z':
        ten_mean = torch.mean(ten)
        ten_std, ten_mean = torch.std_mean(ten, ten.size())
        norm_ten = (norm_ten - ten_mean) / ten_std
    else:
        raise NotImplementedError('Unknown norm_type: ' + norm_type)

    # make thresholds
    threshs = torch.linspace(0.1, 0.9, 9)
    thresh_tens = [(norm_ten < thresh) for thresh in threshs]

    max12 = []
    min12 = []
    med12 = []
    max13 = []
    min13 = []
    med13 = []
    max23 = []
    min23 = []
    med23 = []

    for thresh_ten in thresh_tens:
        n = ten.size(2)
        thresh_12 = torch.zeros(n)
        for k in range(n):
            thresh_12[k] = thresh_ten[:, :, k].sum()
        tmin, tmed, tmax = three_samp(thresh_12)
        min12.append(tmin)
        med12.append(tmed)
        max12.append(tmax)

        n = ten.size(1)
        thresh_13 = torch.zeros(n)
        for k in range(n):
            thresh_13[k] = thresh_ten[:, k, :].sum()
        tmin, tmed, tmax = three_samp(thresh_13)
        min13.append(tmin)
        med13.append(tmed)
        max13.append(tmax)

        n = ten.size(0)
        thresh_23 = torch.zeros(n)
        for k in range(n):
            thresh_23[k] = thresh_ten[k, :, :].sum()
        tmin, tmed, tmax = three_samp(thresh_23)
        min23.append(tmin)
        med23.append(tmed)
        max23.append(tmax)

    # Assign ranges
    to_append = [
        max12,
        min12,
        med12,
        max13,
        min13,
        med13,
        max23,
        min23,
        med23
    ]

    idx = 22
    trange = len(thresh_tens)
    for x in to_append:
        feat[idx:idx + trange] = torch.tensor(x)
        idx += trange

    assert(idx == 103)

    corr_12 = corr_pairs(norm_ten, 2)
    corr_13 = corr_pairs(norm_ten, 1)
    corr_23 = corr_pairs(norm_ten, 0)
    feat[103], feat[104], feat[105] = three_samp_alt(corr_12)
    feat[106], feat[107], feat[108] = three_samp_alt(corr_13)
    feat[109], feat[110], feat[111] = three_samp_alt(corr_23)

    # exclude rank features if we have them disabled
    if not calculate_rank:
        feat = torch.concat([feat[:13], feat[22:]])

    if torch.isnan(feat).sum() != 0:
        print('Dumping features:')
        for i in range(len(feat)):
            print(f'{FEATURE_NAMES[i]}: {feat[i]}')
        raise RuntimeError('NaN feature found')

    return feat


def rank_thresh(ten, cutoff=0.9):
    sigs = torch.linalg.svdvals(ten)
    ranks = []
    sqs = sigs#torch.square(sigs)
    totals = sqs.sum(axis=1)
    for i in range(sigs.size(0)):
        total = 0
        thresh = cutoff * totals[i]
        for k in range(sigs.size(1)):
            if total >= thresh:
                break
            total += sqs[i, k]
        ranks.append(k)
    return torch.tensor(ranks)

# norm_type can either be min-max (default) or z
def extract_features_r90(ten, norm_type='min-max', cutoff=0.9):
    feat = torch.empty(112)
    feat[:] = float('nan')

    # dimensions i, j, k
    feat[0], feat[1], feat[2] = ten.shape

    # nnz
    feat[3] = (ten >= 0.00001).sum()

    # get nnz per slice along (1,2)-mode slices
    feat[4], feat[5], feat[6] = three_samp(torch.count_nonzero(ten, (0,1)))
    # get nnz per slice along (1,3)-mode slices
    feat[7], feat[8], feat[9] = three_samp(torch.count_nonzero(ten, (0,2)))
    # get nnz per slice along (2,3)-mode slices
    feat[10], feat[11], feat[12] = three_samp(torch.count_nonzero(ten, (1,2)))

    # get rank along (1,2)-mode slices
    m12_ranks = rank_thresh(torch.transpose(ten, 0, 2), cutoff=cutoff)
    feat[13], feat[14], feat[15] = three_samp(m12_ranks)

    # get rank along (1,3)-mode slices
    m13_ranks = rank_thresh(torch.transpose(ten, 0, 1), cutoff=cutoff)
    feat[16], feat[17], feat[18] = three_samp(m13_ranks)

    # get rank along (2,3)-mode slices
    m23_ranks = rank_thresh(ten, cutoff=cutoff)
    feat[19], feat[20], feat[21] = three_samp(m23_ranks)

    # normalize tensor
    norm_ten = torch.clone(ten).detach()
    if norm_type == 'min-max':
        min_ten = torch.min(ten)
        max_ten = torch.max(ten)
        norm_ten = (norm_ten - min_ten) / (max_ten - min_ten)
    elif norm_type == 'z':
        ten_mean = torch.mean(ten)
        ten_std, ten_mean = torch.std_mean(ten, ten.size())
        norm_ten = (norm_ten - ten_mean) / ten_std
    else:
        raise NotImplementedError('Unknown norm_type: ' + norm_type)

    # make thresholds
    threshs = torch.linspace(0.1, 0.9, 9)
    thresh_tens = [(norm_ten < thresh) for thresh in threshs]

    max12 = []
    min12 = []
    med12 = []
    max13 = []
    min13 = []
    med13 = []
    max23 = []
    min23 = []
    med23 = []

    for thresh_ten in thresh_tens:
        n = ten.size(2)
        thresh_12 = torch.zeros(n)
        for k in range(n):
            thresh_12[k] = thresh_ten[:, :, k].sum()
        tmin, tmed, tmax = three_samp(thresh_12)
        min12.append(tmin)
        med12.append(tmed)
        max12.append(tmax)

        n = ten.size(1)
        thresh_13 = torch.zeros(n)
        for k in range(n):
            thresh_13[k] = thresh_ten[:, k, :].sum()
        tmin, tmed, tmax = three_samp(thresh_13)
        min13.append(tmin)
        med13.append(tmed)
        max13.append(tmax)

        n = ten.size(0)
        thresh_23 = torch.zeros(n)
        for k in range(n):
            thresh_23[k] = thresh_ten[k, :, :].sum()
        tmin, tmed, tmax = three_samp(thresh_23)
        min23.append(tmin)
        med23.append(tmed)
        max23.append(tmax)

    # Assign ranges
    to_append = [
        max12,
        min12,
        med12,
        max13,
        min13,
        med13,
        max23,
        min23,
        med23
    ]

    idx = 22
    trange = len(thresh_tens)
    for x in to_append:
        feat[idx:idx + trange] = torch.tensor(x)
        idx += trange

    assert(idx == 103)

    corr_12 = corr_pairs(norm_ten, 2)
    corr_13 = corr_pairs(norm_ten, 1)
    corr_23 = corr_pairs(norm_ten, 0)
    feat[103], feat[104], feat[105] = three_samp_alt(corr_12)
    feat[106], feat[107], feat[108] = three_samp_alt(corr_13)
    feat[109], feat[110], feat[111] = three_samp_alt(corr_23)

    if torch.isnan(feat).sum() != 0:
        print('Dumping features:')
        for i in range(len(feat)):
            print(f'{FEATURE_NAMES[i]}: {feat[i]}')
        raise RuntimeError('NaN feature found')
    return feat