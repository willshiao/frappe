function model = BayesCPWrapper(X)
DIM = size(X);
%% Random missing values
ObsRatio = 1;    % observation rate (i.e., 1-missing rate)
Omega = randperm(prod(DIM)); 
Omega = Omega(1:round(ObsRatio*prod(DIM)));
O = zeros(DIM); 
O(Omega) = 1;

%% Generate observation tensor Y
Y = X;

%% Run BayesCP

fprintf('------Bayesian CP Factorization---------- \n');
tic
[model] = BayesCP(X, 'obs', O, 'init', 'ml', 'maxRank', max(DIM), 'dimRed', 1, 'tol', 1e-5, 'maxiters', 200);
t_total = toc;
