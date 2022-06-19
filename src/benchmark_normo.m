clc
clear all

%load dataset
dset_path = '../data/diverse-ten_500.h5';
n_samp = h5read(dset_path, '/n_tens');
ranks = h5read(dset_path, '/ranks');
fprintf('Loading %d samples\n', n_samp)

% load NORMO functions
addpath(genpath('NORMO'));

normo_e = zeros(1, n_samp);
normo_b = zeros(1, n_samp);
avg_ts_e = zeros(1, n_samp);
avg_ts_b = zeros(1, n_samp);

% if a run was interrupted, start at this index:
start_at = 0;
for i=n_samp-1:-1:start_at
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

    %apply NORMO
    R_e=[]; t_e=[];
    R_b=[]; t_b=[];
    for seed=[0,66]
        max_rank = double(ranks(k)*2);
        %apply exhaustive search NORMO
        [nredndantcomps,t_e(end+1)]=e_NORMO(T,max_rank,0.7,seed);
        %get number of components for which there were redundancy
        Rs_with_redundancy=find(nredndantcomps>0);
        %get the maximal number of components R such there is no redundancy
        %when decomposing with R or with <R components but there is redundancy
        %when decomposing with R+1 components
        if numel(Rs_with_redundancy) == 0
            R_e(end+1)=100;
        else
            R_e(end+1)=Rs_with_redundancy(1)-1;
        end
        
        %apply binary search NORMO
        [R_b(end+1),t_b(end+1)]=b_NORMO(T,13,0.7,seed);
    end
    R_e=mode(R_e); avg_t_e=mean(t_e);std_t_e=std(t_e);
    R_b=mode(R_b); avg_t_b=mean(t_b);std_t_b=std(t_b);

    % store results
    normo_e(k) = R_e;
    normo_b(k) = R_b;
    avg_ts_e(k) = avg_t_e;
    avg_ts_b(k) = avg_t_b;    
end

save('../data/diverse_normo.mat', 'normo_b', 'normo_e', "avg_ts_e", "avg_ts_b");
disp('Done!')
