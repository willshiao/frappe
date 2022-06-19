function [Core, FACT, alpha,lambda, nlogP, VarExpl] = ARDTUCKER(X, noc, varargin)

% ARD Tucker
%
% Written by Morten M�rup
%
% When using the algorithm please cite:
%   Morten M�rup and Lars Kai Hansen "Automatic Relevance Determination for
%   Multi-way models"
%
% Model:
%   X=Core x_1 FACT{1} x_2 FACT{2} x_3 ... x_N FACT{N}
%
% Usage:
% [Core, FACT, alpha,lambda, nlogP, VarExpl] = ARDTUCKER(X, noc, varargin)
%
% Input:
% X             n-way array to decompose
% noc           vector of potential number of components for each mode
% opts.         Struct containing:
%       SNR          Signal to Noise Ratio used to estimate sigma^2 (default 0)
%       method       'Sparse' or 'Ridge' ('Sparse' default corresponding to L1 regularization)
%       constrFACT       constrFACT(n)=1 --> FACT{n} non-negative, constrFACT(n)=0 -->
%                    FACT{n} unconstrFACTained
%       constrCore   1: Core non-negative, 0: Core unconstrFACTained
%       FACT         initial solution (optional) (see also output)
%       Core         initial solution (optional) (see also output)
%       constFACT    constFACT(i)=0  FACT{i} updated, else FACT{i} not updated
%                    but kept constant (default 0)
%       constCore    1: Core array kept fixed (if initialized by a diagonal tensor this corresponds to CP optimization)
%                    0: Core estimated
%       maxiter      maximum number of iterations
%       conv_crit    The convergence criteria (defauld 10^-9 relative change in negLogP)
%       noARDiter    number of initial iterations without ARD (default 25)
%
% Output:
% Core          Core array 
% FACT          cell array: FACT{n} is the factors found for the n'th
%               modality
% alpha         cell array of regularization strengths in components given
%               in FACT
% lambda        regularization strength on Core
% nlogP         negative log likelihood at final iteration
% varexpl       Percent variation explained by the model
% Revision:
%  21 july 2014 bug that was introduced fixed where components were
%               potentially pruned prematurely due to criteria on line 207 now replaced
%               with line 208. maxiter further set to 10000 instead of 500 as default.
%  22 july 2014 sigma_sq in line 208 changed to scale
       
warning('off','MATLAB:dispatcher:InexactMatch')
addpath(genpath('N-way-tools'));
addpath(genpath('Optim-tools'));

Nx=ndims(X);
N=size(X);

% Identify missing values
W=ones(size(X));
missing=isnan(X);
nr_missing=sum(missing(:));
W(missing)=0;
X(missing)=0;
SST=sum(X(:).^2);

if nargin>=3, opts = varargin{1}; else opts = struct; end

% Extract algorithm settings
conv_crit=mgetopt(opts,'conv_crit',10^-9);
maxiter=mgetopt(opts,'maxiter',10000);
constrFACT=mgetopt(opts,'constrFACT',zeros(1,Nx));
constrCore=mgetopt(opts,'constrCore',0);
constFACT=mgetopt(opts,'constFACT',zeros(Nx,1));
constCore=mgetopt(opts,'constCore',0);
method=mgetopt(opts,'method','Sparse');
FACT=mgetopt(opts,'FACT',[]);
Core=mgetopt(opts,'Core',[]);
SNR=mgetopt(opts,'SNR',0); 
noARDiter=mgetopt(opts,'noARDiter',25);
scale=SST/(prod(N)-nr_missing);
sigma_sq=SST/((1+10^(SNR/10))*(prod(N)-nr_missing));
for i=1:Nx
    alpha{i}=sqrt(sigma_sq)*1e-12*ones(noc(i),1); % Start with practically zero value of regularization
end
alpha=mgetopt(opts,'alpha',alpha);
lambda=mgetopt(opts,'lambda',sqrt(sigma_sq)*1e-12);

% Random initialisation
if isempty(FACT) 
    for i=1:Nx       
        if constrFACT(i)
            FACT{i}=rand(N(i),noc(i));
        else
            FACT{i}=randn(N(i),noc(i));
        end            
    end
end
if isempty(Core)
    if constCore
        Core=core_init(noc);
    else
        if constrCore
            Core=rand(noc);
        else
            Core=randn(noc);
        end
    end
end

% Set initial parameters
mu_c=1;
mu=ones(1,Nx);
tol=1e-9;
iter=0;
dnlogP=inf;
nlogP=inf;
constTerm=0;
terminate=0;
Rec=reconstructTucker(Core,FACT);
X(missing)=Rec(missing);
cost_ls=0.5*sum(W(:).*(X(:)-Rec(:)).^2);
corecost=0;
nrmCore=sum(abs(Core(:)));
cpu_time=cputime;

% Display algorithm
disp([' '])
disp(['ARD Tucker Decomposition based on ' method '-regression' ])
disp(['A ' num2str(noc) ' component model will be fitted']);
disp([ num2str(nr_missing/prod(N)) ' pct. of data treated as missing values '])
disp(['To stop algorithm press control C'])
disp([' ']);
dheader = sprintf('%12s | %12s | %12s | %12s | %12s ','Iteration','Expl. var.','nlogP','Delta nlogP','Time');
dline = sprintf('-------------+--------------+--------------+--------------+--------------+');

while abs(dnlogP)>=conv_crit*abs(nlogP) & iter<maxiter & ~terminate 

        if mod(iter,100)==0
             disp(dline); disp(dheader); disp(dline);
        end
        iter=iter+1;
        
        cpu_time_old=cpu_time;
        nlogP_old=nlogP;
        regNorm=0;
        regCost=0;

        % Estimate Core   
        if ~constCore
            if strcmp(method,'Sparse')
                if nr_missing==0
                    [Core,mu_c]=estimateSparseCore(X,Core,FACT,sigma_sq*lambda,constrCore,mu_c,50);
                    Rec=reconstructTucker(Core,FACT);        
                else
                    [Core,Rec,mu_c]=estimateSparseCoreMis(X,Rec,W,Core,FACT,lambda*sigma_sq,constrCore,mu_c,5);     
                end
                nrmCore=sum(abs(Core(:)));
                if iter>noARDiter
                    lambda=prod(noc)/(nrmCore+tol);
                end
                corecost=-prod(noc)*log(lambda)+lambda*nrmCore-log(tol)+lambda*tol;
            else
                if nr_missing==0
                    [Core,mu_c]=estimateRidgeCore(X,Core,FACT,sigma_sq*lambda,constrCore,mu_c,50);
                    Rec=reconstructTucker(Core,FACT);        
                else
                    [Core,Rec,mu_c]=estimateRidgeCoreMis(X,Rec,W,Core,FACT,lambda*sigma_sq,constrCore,mu_c,5);                        
                end
                nrmCore=sum(Core(:).^2);
                if iter>noARDiter    
                    lambda=prod(noc)/(nrmCore+tol);
                end
                corecost=-0.5*prod(noc)*log(lambda)+0.5*lambda*nrmCore-0.5*log(tol)+0.5*lambda*tol;
            end
            X(missing)=Rec(missing);   
        end

        % Estimate FACT(1:end)
        for i=1:Nx                        
            if ~constFACT(i) & ~terminate
                % Remove FACT{i} from Rec
                if rank(FACT{i})>=size(FACT{i},2) && Nx>2
                       Rec=tmult(Rec,pinv(FACT{i}),i);
                else
                        t=1:Nx;
                        t(i)=[];
                        Rec=Core;
                        for tt=t
                           Rec=tmult(Rec,FACT{tt},tt); 
                        end
                end
                Xi=matrizicing(X,i);
                CFi=matrizicing(Rec,i);
                XCF=Xi*CFi';
                C=CFi*CFi';

                if strcmp(method,'Sparse')                   
                    [FACT{i},mu(i),cost_ls,cost_reg]=gbsc(XCF,C,FACT{i},mu(i),sigma_sq*alpha{i}',constrFACT(i),100);
                else
                    [FACT{i},mu(i),cost_ls,cost_reg]=gbrc(XCF,C,FACT{i},mu(i),sigma_sq*alpha{i}',constrFACT(i),100);                    
                end
                CC=FACT{i}'*FACT{i};
                Rec=tmult(Rec,FACT{i},i);
                X(missing)=Rec(missing);

                % Remove pruned components
                %ind=find(diag(CC)<sqrt(sigma_sq)*tol);
                ind=find((diag(CC).*mean(CFi.^2,2))<scale*tol);
                if length(ind)>0 & iter>noARDiter
                    disp(['removing ' num2str(length(ind)) ' component(s) from mode ' num2str(i)]);
                    FACT{i}(:,ind)=[];
                    CC(ind,:)=[];
                    CC(:,ind)=[]; 
                    q=noc;
                    q(i)=[];
                    if strcmp(method,'Sparse')
                        constTerm=constTerm-length(ind)*N(i)*log(N(i)/tol)-length(ind)*log(tol)+tol*length(ind)*N(i)/tol-prod(q)*length(ind)*log(lambda);
                    else
                        constTerm=constTerm-0.5*length(ind)*N(i)*log(N(i)/tol)-0.5*length(ind)*log(tol)+0.5*tol*length(ind)*N(i)/tol-0.5*prod(q)*length(ind)*log(lambda);
                    end
                    alpha{i}(ind)=[];
                    Core=matrizicing(Core,i);
                    Core(ind,:)=[];
                    noc(i)=noc(i)-length(ind);
                    Core=unmatrizicing(Core,i,noc);
                    if strcmp(method,'Sparse')
                        corecost=-prod(noc)*log(lambda)+lambda*nrmCore-log(tol)+lambda*tol;
                    else
                        corecost=-0.5*prod(noc)*log(lambda)+0.5*lambda*nrmCore-0.5*log(tol)+0.5*lambda*tol;
                    end                    
                    if noc(i)==0
                        terminate=1;
                        disp(['All components removed in mode ' num2str(i) ' hence algorithm terminated'])
                    else
                        Rec=reconstructTucker(Core,FACT);        
                    end
                end

                % Re-estimate alpha{i}
                if ~terminate
                    if strcmp(method,'Sparse')
                        nrmFACT=sum(abs(FACT{i}));
                        if iter>noARDiter
                            alpha{i}=N(i)./(nrmFACT'+tol);
                        end
                        regCost=regCost+nrmFACT*alpha{i};
                        regNorm=regNorm-N(i)*sum(log(alpha{i}))-noc(i)*log(tol)+tol*sum(alpha{i});
                    else
                        nrmFACT=sum(FACT{i}.^2);
                        if iter>noARDiter
                           alpha{i}=N(i)./(nrmFACT'+tol);
                        end
                        regCost=regCost+0.5*nrmFACT*alpha{i};
                        regNorm=regNorm-0.5*N(i)*sum(log(alpha{i}))-0.5*noc(i)*log(tol)+0.5*tol*sum(alpha{i});
                    end                   
                end                
            end         
        end     
        if ~terminate
            X(missing)=Rec(missing);
            cost_ls=0.5*sum((X(:)-Rec(:)).^2);
            
            % Evaluate negative Log likelihood
            nlogP=cost_ls/sigma_sq+0.5*(prod(N)-nr_missing)*log(sigma_sq)+regCost+regNorm+corecost...
               +constTerm;
            dnlogP=nlogP_old-nlogP;    
            VarExpl=(SST-2*cost_ls)/SST;
            if rem(iter,5)==0
                cpu_time=cputime;
                disp(sprintf('%12.0f | %12.4f | %12.4f | %12.4e | %12.4e ',iter, VarExpl,nlogP,dnlogP/abs(nlogP),cpu_time-cpu_time_old));
            end
        end
end
if ~terminate
    disp(sprintf('%12.0f | %12.4f | %12.4f | %12.4e ',iter, VarExpl,nlogP,dnlogP/abs(nlogP)))
end

% -------------------------------------------------------------------------
% Parser for optional arguments
function var = mgetopt(opts, varname, default, varargin)
if isfield(opts, varname)
    var = getfield(opts, varname); 
else
    var = default;
end
for narg = 1:2:nargin-4
    cmd = varargin{narg};
    arg = varargin{narg+1};
    switch cmd
        case 'instrset',
            if ~any(strcmp(arg, var))
                fprintf(['Wrong argument %s = ''%s'' - ', ...
                    'Using default : %s = ''%s''\n'], ...
                    varname, var, varname, default);
                var = default;
            end
        otherwise,
            error('Wrong option: %s.', cmd);
    end
end