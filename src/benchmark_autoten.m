clc
clear all

%load dataset
dset_path = '../data/diverse-ten_500.h5';
n_samp = h5read(dset_path, '/n_tens');
ranks = h5read(dset_path, '/ranks');
fprintf('Loading %d samples\n', n_samp)

% load NORMO functions
addpath(genpath('AutoTen'));

autoten_f = zeros(n_samp, 1);
autoten_l = zeros(n_samp, 1);
autoten_timings = zeros(n_samp, 1);

% if a run was interrupted, start at this index:
start_at = 0;
for i=start_at:n_samp-1
    k = i+1; % for MATLAB-style indexing
    T = double(h5read(dset_path, sprintf("/%d", i)));
    ndim = numel(size(T));
    T = permute(T, [ndim:-1:1]);
    if numel(size(T)) < 3
        fprintf("Reshape tensor #%d", k)
        T = reshape(T, [1 size(T)]);
        ndim = numel(size(T));
    end

    fprintf("==================== Working on tensor #%d/%d ====================\n", i, n_samp-1)
    fprintf('Tensor size: ')
    disp(size(T))

    %apply AutoTen
    Fmax = double(2*ranks(k));
    tStart = tic;
    [Fac, c, F_est,loss] = AutoTen(tensor(T), Fmax, 1);
    autoten_timings(k) = toc(tStart);
    autoten_f(k) = F_est;

    if mod(k, 25) == 0
        fprintf('Saving checkpoint at iteration %d', k)
        save('../data/ckpt_diverse_autoten.mat', 'autoten_f', 'autoten_timings');
    end
end

save('../data/diverse_autoten.mat', 'autoten_f', 'autoten_timings');
disp('Done!')