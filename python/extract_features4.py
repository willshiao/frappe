import itertools
import torch

CALC_FEAT_NAMES = []
SHARED_3D_MODEL = None
HAVE_FEATS = False

def get_feature_names():
    if CALC_FEAT_NAMES:
        return CALC_FEAT_NAMES
    return None

def three_samp(vec):
    filt_vec = vec[torch.isfinite(vec)]
    if filt_vec.numel() == 0:
        return 0, 0, 0
    return torch.min(filt_vec), torch.median(filt_vec), torch.max(filt_vec)

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

def add_three_corr_feat(feat_name):
    if not HAVE_FEATS:
        CALC_FEAT_NAMES.extend([
            'min_' + feat_name,
            'median_' + feat_name,
            'max_' + feat_name
        ])

def add_feat_name(feat_name):
    if not HAVE_FEATS:
        CALC_FEAT_NAMES.append(feat_name)

def add_feat_names(feat_names):
    if not HAVE_FEATS:
        CALC_FEAT_NAMES.extend(feat_names)

def finalize_feats():
    global HAVE_FEATS
    HAVE_FEATS = True

# norm_type can either be min-max (default) or z
def extract_features4d(ten, norm_type='min-max'):
    ndim = len(ten.size())
    feat = []

    # dimensions
    feat.extend(ten.size())
    add_feat_names([f'dim{x}' for x in range(ndim)])

    # nnz
    feat.append((ten >= 0.00001).sum())
    add_feat_name('nnz')

    perms = itertools.combinations(range(ndim), ndim - 1)
    for perm in perms:
        feat.extend(three_samp(torch.count_nonzero(ten, perm)))
        add_three_corr_feat(f'nnz_along_({",".join(map(str, perm))})-mode_slices')

    # Skip rank-related features for now
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

    for idx, thresh_ten in enumerate(thresh_tens):
        for dim in range(ndim):
            n = ten.size(dim)
            thresh_area = torch.zeros(n)

            for k in range(n):
                thresh_area[k] = thresh_ten.select(dim, k).sum()
            feat.extend(three_samp(thresh_area))
            add_three_corr_feat(f'nnz_over_{dim}-mode_cubes_with_thresh={round(float(threshs[idx]), 1)}')

    for dim in range(ndim):
        feat.extend(three_samp(corr_pairs(norm_ten, dim)))
        add_three_corr_feat(f'corr_over_{dim}-mode_cubes')

    finalize_feats()
    return torch.tensor(feat, dtype=torch.float64)
