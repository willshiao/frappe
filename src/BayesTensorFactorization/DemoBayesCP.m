% A demo of Bayesian CP Factorization for Low-Rank Tensor Completion
% Author:  Qibin Zhao   2013

close all;
randn('state',1); rand('state',1); %#ok<RAND>
addpath(genpath('BayesCP'))
%% Generate a low-rank tensor
% Dimensions
DIM = [40,40,40];  % tensor size
R = 3; %  true CP  rank
lambda = ones(1,R);

DataType = 2;  % Data type  1: random data  2: deterministic signals
Z = cell(length(DIM),1);   
if DataType ==1
    for m=1:length(DIM)
          Z{m} =  gaussSample(zeros(R,1), eye(R), DIM(m));
%         Z{m} =  gaussSample(zeros(DIM(m),1), eye(DIM(m)), R)';
    end
end
if DataType == 2
    for m=1:length(DIM)
        temp = linspace(0, m*2*pi, DIM(m));
        part1 = [sin(temp);  cos(temp); square(linspace(0, 15*pi, DIM(m)))]';
        part2 = gaussSample(zeros(DIM(m),1), eye(DIM(m)), R-size(part1,2))';
        Z{m} = [part1 part2];
        Z{m} = Z{m}(:,1:R);
    end
end

% generate tensor from factor matrices
X = double(ktensor(lambda',Z));
TrueRank = max(cellfun(@(x) rank(x), Z));

%% Random missing values
ObsRatio = 0.1;    % observation rate (i.e., 1-missing rate)
Omega = randperm(prod(DIM)); 
Omega = Omega(1:round(ObsRatio*prod(DIM)));
O = zeros(DIM); 
O(Omega) = 1;

%% Add noise
SNR = 20;   % Noise level
sigma2 = var(X(:))*(1/(10^(SNR/10)));
GN =  sqrt(sigma2)*randn(DIM);

%% Generate observation tensor Y
Y = X  + GN;
Y = O.*Y;
SNR = 10*log10(var((O(:).*X(:))) / var((O(:).*GN(:))));

%% Run BayesCP

fprintf('------Bayesian CP Factorization---------- \n');
tic
[model] = BayesCP(Y, 'obs', O, 'init', 'ml', 'maxRank', max(DIM), 'dimRed', 1, 'tol', 1e-5, 'maxiters', 200, 'verbose', 3);
t_total = toc;

% Performance evaluation
X_hat = double(ktensor(model.Z));
err = X_hat(:) - X(:);
rmse = sqrt( mean(err.^2));
rrse = sqrt(sum(err.^2)/sum(X(:).^2));

% Report results
fprintf('\n------------Bayesian CP Factorization-----------------------------------------------------------------------------------\n')
fprintf('Observation ratio = %g, SNR = %g, TrueRank=%d\n', ObsRatio, SNR, TrueRank);
fprintf('RRSE = %g, RMSE = %g, estimated rank = %d, \nEstimated noise Sigma^2 = %g, time = %g\n', ...
    rrse, rmse, max(model.TrueRank), model.beta^(-1), t_total);
fprintf('--------------------------------------------------------------------------------------------------------------------------\n')

%% Visualization of data and results
plotYXS(Y, X_hat);
factorCorr = plotFactor(Z,model.Z);


%% Classical CP
% tic
% P = cp_als(tensor(Y),R);
% toc
% X_hat = double(P);
% err = X_hat(:) - X(:);
% rmse = sqrt( mean(err.^2));
% rrse = sqrt(sum(err.^2)/sum(X(:).^2));
% % Report results
% fprintf('\n-------------CP-ALS------------------------------------------\n')
% fprintf('RRSE = %g, RMSE = %g, \n', rrse, rmse);
% fprintf('-------------------------------------------------------------\n')
% 
% factorCorr = plotFactor(Z,P.U);


