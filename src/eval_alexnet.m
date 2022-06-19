load('../data/trained_alexnet_layer-10.mat')
T = double(data);
autoten_pred = [];
ard_pred = [];
normo_pred = [];

addpath(genpath('AutoTen'));
addpath(genpath('NORMO'));
addpath(genpath('ARDTUCKER'));

i = 1;
k = 1;
max_rank = double(100);
Fmax = max_rank;

% AutoTen
[Fac, c, F_est,loss] = AutoTen(tensor(T), max_rank, 1);
autoten_pred(i) = F_est;

% NORMO
R_e=[]; t_e=[];
R_b=[]; t_b=[];
for seed=[0,66,99,132]
    %apply exhaustive search NORMO
    [nredndantcomps, t_e(end+1)] = e_NORMO(T, max_rank, 0.7, seed);
    %get number of components for which there were redundancy
    Rs_with_redundancy=find(nredndantcomps>0);
    %get the maximal number of components R such there is no redundancy
    %when decomposing with R or with <R components but there is redundancy
    %when decomposing with R+1 components
    R_e(end+1)=Rs_with_redundancy(1)-1;
end
R_e=mode(R_e); avg_t_e=mean(t_e);std_t_e=std(t_e);
normo_pred(k) = R_e;

% ARD
opts.noARDiter=25;
opts.method='Dense';
opts.constrFACT=[0 0 0 0];
opts.constrCore=0;
opts.maxiter = 1000;
G = zeros(Fmax,Fmax,Fmax,Fmax);for f=1:Fmax,G(f,f,f,f) = 1;end
opts.Core = G;
opts.constCore = 1;
[Core_est, FACT_est, alpha_est,varexpl] = ...
        ARDTUCKER(T, [Fmax Fmax Fmax Fmax],opts);
ard_pred(k) = size(Core_est,1);
