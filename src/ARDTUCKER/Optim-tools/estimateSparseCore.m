%----------------------------------------------------------
function [Core,mu_c]=estimateSparseCore(X,Core,FACT,lambda,CoreConstr,mu_c,iter);
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
dCore=inf;
CC=Core;
XC=X;
for i=1:length(FACT)
    C{i}=FACT{i}'*FACT{i};
    XC=tmult(XC,FACT{i}',i);
    CC=tmult(CC,C{i},i);
    F{i}=diag(C{i});
end
Hes=outerprod(F)+eps;
k=1;
while k<iter && dCore>tol
    Gr=CC-XC;
    cost_old=0.5*sum(CC(:).*Core(:))-sum(XC(:).*Core(:))+lambda*sum(abs(Core(:)));
    stop=0;  
    Core_old=Core;
    while ~stop
        Core=Core_old-mu_c*Gr./Hes;
        Core(abs(Core).*Hes<mu_c*lambda)=0;
        Core=Core-mu_c*lambda*sign(Core)./Hes;
        if CoreConstr
           Core(Core<0)=0; 
        end
        CC=Core;
        for i=1:length(FACT)
           CC=tmult(CC,C{i},i);
        end
        cost=0.5*sum(CC(:).*Core(:))-sum(XC(:).*Core(:))+lambda*sum(abs(Core(:)));
        if cost<=cost_old+eps 
            mu_c=mu_c*1.2;
            dCore=sum((Core(:)-Core_old(:)).^2)/(sum(Core(:).^2)+eps);
            stop=1;
        else
            mu_c=mu_c/2;
        end
    end
    k=k+1;
end
