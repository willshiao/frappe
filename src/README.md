# MATLAB Baselines

This directory contains the code for the MATLAB baseline methods.

Note that the majority of the code comes from the original authors. You must first generate the data with `python/make_matlab_test.py`.

The baselines are:

- `ARD-Tucker` run with `benchmark_ard.m`
- `AutoTen`: run with `benchmark_autoten.m`
- `NORMO`: run with `benchmark_normo.m`

Note that `AutoTen` has been slightly modified to only use the Frobenius norm method. This is because of negative values in the tensors (as described in the paper). `AutoTenOrig.m` contains the original `AutoTen` code. AutoTen's `efficient_corcondia.m` and NORMO's `NORMO/e_normo.m` were modified to work with 4D tensors.

AutoTen requires the Tensor Toolbox, and NORMO requires the n-way Toolbox. ARD-Tucker includes all of its dependencies.

`eval_alexnet.m` evaluates the methods on the AlexNet downstream task, as described in the paper. `eval_known.m` evaluates the methods on the tensor with known rank, as described in the paper.