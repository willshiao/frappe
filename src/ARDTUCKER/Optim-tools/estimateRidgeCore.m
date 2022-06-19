%----------------------------------------------------------
function [Core,mu_c]=estimateRidgeCore(X,Core,FACT,lambda,CoreConstr,mu_c,iter);
% Update the Core in the Tucker model L1 regularized by lambda
%
%   Input
%       X       n-way array
%       Rec     Reconstructed n-way array according to the Tucker model
%       W       n-way array indicating missing values, 0 where data
%               missing, 1 where present
%       Core    Core in Tucker model
%       FACT    cell array containing the factors of each mode of the
%               Tucker model
%       lambda  L1 regularization strength
%       CoreConstr  constraint on Core, 1: non-negative, 0: unconstrained
%       mu_c        step size
%       iter    maximal number of iterations
%
%   Output
%       Core    updated Core array in Tucker model
%       Rec     updated Reconstructed n-way array
%       mu_c    step size
tol=1e-9;
CC=Core;
dCore=inf;
Hes=ones(size(Core));
XC=X;
for i=1:length(FACT)
    C{i}=FACT{i}'*FACT{i};
    XC=tmult(XC,FACT{i}',i);
    CC=tmult(CC,C{i},i);
    F{i}=diag(C{i});
end
Hes=outerprod(F)+lambda;
k=1;
while k<iter && dCore>tol
    Gr=CC-XC+lambda*Core;
    cost_old=0.5*sum(CC(:).*Core(:))-sum(XC(:).*Core(:))+0.5*lambda*sum(Core(:).^2);
    stop=0;  
    Core_old=Core;
    while ~stop
        Core=Core_old-mu_c*Gr./Hes;
        if CoreConstr
           Core(Core<0)=0; 
        end
        CC=Core;
        for i=1:length(FACT)
           CC=tmult(CC,C{i},i);
        end
        cost=0.5*sum(CC(:).*Core(:))-sum(XC(:).*Core(:))+0.5*lambda*sum(Core(:).^2);
        if cost<=cost_old+eps 
            mu_c=mu_c*1.2;
            stop=1;
            dCore=sum((Core(:)-Core_old(:)).^2)/sum(Core(:).^2);
        else
            mu_c=mu_c/2;
        end
    end
    k=k+1;
end

