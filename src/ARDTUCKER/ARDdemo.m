% Analysis of a Synthetic example using ARD Tucker
% The real datasets used in article can be downloaded from:
%   www.models.kvl.dk/research/data/

N=[30, 40,50];
noc=[3 4 5];

% Generate data
addpath(genpath('N-way-tools'));
for k=1:length(N)
    FACT{k}=randn(N(k),noc(k));
end
Core=randn(noc);
Rec=reconstructTucker(Core,FACT);

% Add noise such that SNR=0dB
S=norm(Rec(:),'fro')^2;
X=Rec+sqrt(S/prod(N))*randn(N);

% Set algorithm parameters
opts.noARDiter=25;
opts.method='Sparse';
opts.constrFACT=[0 0 0];
opts.constrCore=0;
[Core_est, FACT_est, alpha_est,varexpl] = ARDTUCKER(X, [10 10 10],opts);
