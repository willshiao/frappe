clc
clear all

%load dataset
dset_path = '../data/diverse-ten_500.h5';
n_samp = h5read(dset_path, '/n_tens');
ranks = h5read(dset_path, '/ranks');
fprintf('Loading %d samples\n', n_samp)

% load ARD functions
addpath(genpath('ARDTUCKER'));

baseline_ranks = zeros(n_samp,1);
tucker_timings = zeros(n_samp, 1);

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

    % apply ARD
    Fmax = double(2*ranks(k));

    % set options
    try
        opts.noARDiter=50;
        opts.method='Dense';
        opts.constrFACT=[0 0 0];
        opts.constrCore=0;
        opts.maxiter = 1000;
        G = zeros(Fmax,Fmax,Fmax);for f=1:Fmax,G(f,f,f) = 1;end
        opts.Core = G;
        opts.constCore = 1;
        tic
        [Core_est, FACT_est, alpha_est,varexpl] = ...
                ARDTUCKER(T, [Fmax Fmax Fmax],opts);
        tucker_timings(k) = toc;
        baseline_ranks(k) = size(Core_est,1);
    
%         if mod(k, 25) == 0
%             fprintf('Saving checkpoint at iteration %d\n', k)
%             save('../data/ard_run.mat', 'baseline_ranks', 'tucker_timings');
%         end
    catch err
        warning(['Error calculating decomposition for index ', k, ':' err.message]);
        tucker_timings(k) = nan;
        baseline_ranks(k) = nan;
    end
end

save('../data/diverse_ard.mat', 'baseline_ranks', 'tucker_timings');
disp('Done!')