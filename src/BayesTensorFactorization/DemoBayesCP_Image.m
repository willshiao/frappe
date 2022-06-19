
% Bayesian CP factorization for image completion
% Written by Qibin Zhao 2014


% Read image data 
filename='.\TestImages\peppers.bmp';   % image files
ObsRatio = 0.05; % Observation rate 



randn('state',1); rand('state',1); %#ok<RAND>
img = imread(filename);
X = double(img);
DIM = size(X);

Omega = randperm(prod(DIM));
Omega = Omega(1:round(ObsRatio*prod(DIM)));
O = zeros(DIM);
O(Omega) = 1;
Y = O.*X;

% plot images
if 1
    subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.01], [0.01 0.01], [0.01 0.01]);
    row =1; col =2;
    figure;
    subplot(row,col,1);
    imshow(uint8(X));
    subplot(row,col,2);
    imshow(uint8(Y));
    drawnow;
end

% Initialization
TimeCost = zeros(2,1);
SRunCost = zeros(2,1);
RSElist = zeros(2,3);
PSNRlist = zeros(2,1);
SSIMlist = zeros(2,1);
RankEst = zeros(2,1);


%% FBCP for structural images 
if 1
    tStart = tic;
    fprintf('------Bayesian CP factorization---------- \n');
    [model] = BayesCP_Img(Y, 'obs', O, 'init', 'rand', 'maxRank', 100, 'maxiters', 20, ...
        'tol', 1e-4, 'dimRed', 1, 'verbose', 2);
    X_FBCP = double(ktensor(model.Z));
    RSElist(1,1) = perfscore(X_FBCP, X);
    RSElist(1,2) = perfscore(X_FBCP(O==1), X(O==1));
    RSElist(1,3) = perfscore(X_FBCP(O==0), X(O==0));
    
    X_FBCP(O==1) = X(O==1);
    PSNRlist(1) = PSNR_RGB(X_FBCP,X);
    SSIMlist(1) = ssim_index(rgb2gray(uint8(X_FBCP)),rgb2gray(uint8(X)));
    RankEst(1) = model.TrueRank;
    TimeCost(1) = toc(tStart);
    figure; imshow(uint8(X_FBCP)); title('FBCP'); drawnow;    
end

%% FBCP-MP (mixture priors) for natural images
if isempty(strfind(filename,'facade.bmp'))
    tStart = tic;
    fprintf('------Bayesian CP with Mixture Priors---------- \n');
    [model] = BayesCP_MP(Y, 'obs', O, 'init', 'rand', 'maxRank', 100, 'maxiters', 30, ...
        'tol', 1e-4, 'dimRed', 1, 'verbose', 2);
    X_FBCPS = double(ktensor(model.Z));    
    
    RSElist(2,1) = perfscore(X_FBCPS, X);
    RSElist(2,2) = perfscore(X_FBCPS(O==1), X(O==1));
    RSElist(2,3) = perfscore(X_FBCPS(O==0), X(O==0));
    
    
    X_FBCPS(O==1) = X(O==1);
    PSNRlist(2) = PSNR_RGB(X_FBCPS,X);
    SSIMlist(2) = ssim_index(rgb2gray(uint8(X_FBCPS)),rgb2gray(uint8(X)));
    RankEst(2) = model.TrueRank;
    TimeCost(2) = toc(tStart);
    figure; imshow(uint8(X_FBCPS)); title('FBCP-MP'); drawnow;   
end


%%  FBCP-MP (mixture priors)  for low-rank structural images
if strfind(filename,'facade.bmp')
    tStart = tic;
    fprintf('------Bayesian CP with Mixture Priors---------- \n');
    [model] = BayesCP_MP(Y, 'obs', O, 'init', 'rand', 'maxRank', 100, 'maxiters', 30, ...
        'tol', 1e-4, 'dimRed', 1, 'verbose', 2, 'nd', 0.1);
    X_FBCPS = double(ktensor(model.Z));    
    
    RSElist(2,1) = perfscore(X_FBCPS, X);
    RSElist(2,2) = perfscore(X_FBCPS(O==1), X(O==1));
    RSElist(2,3) = perfscore(X_FBCPS(O==0), X(O==0));
    
    
    X_FBCPS(O==1) = X(O==1);
    PSNRlist(2) = PSNR_RGB(X_FBCPS,X);
    SSIMlist(2) = ssim_index(rgb2gray(uint8(X_FBCPS)),rgb2gray(uint8(X)));
    RankEst(2) = model.TrueRank;
    TimeCost(2) = toc(tStart);
    figure; imshow(uint8(X_FBCPS)); title('FBCP-MP'); drawnow;   
end

%%
RankEst
RSElist
PSNRlist
SSIMlist
TimeCost



