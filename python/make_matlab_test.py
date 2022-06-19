from dset import generate_diverse_tens
import h5py

N_SAMPLES = 500

print('generating samples')
tens, ranks = generate_diverse_tens(n=N_SAMPLES, return_tens=True)
with h5py.File(f'../data/diverse-ten_{N_SAMPLES}2.h5', 'w', libver='latest') as f:
    f['ranks'] = ranks
    f['n_tens'] = len(tens)
    for idx, ten in enumerate(tens):
        ten = ten.numpy()
        print(ten.shape)
        f.create_dataset(str(idx), shape=ten.shape, data=ten, chunks=ten.shape, compression='gzip')
print('done!')
