%----------------------------------------------------------
function [Core,Rec,mu_c]=estimateRidgeCoreMis(X,Rec,W,Core,FACT,lambda,CoreConstr,mu_c,iter);

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
%       lambda  L2 regularization strength
%       CoreConstr  constraint on Core, 1: non-negative, 0: unconstrained
%       mu_c        step size
%       iter     maximal number of iterations
%
%   Output
%       Core    updated Core array in Tucker model
%       Rec     updated Reconstructed n-way array
%       mu_c    step size

G=W.*(Rec-X);
for k=1:iter
    Gr=G;
    for i=1:length(FACT)
        Gr=tmult(Gr,FACT{i}',i);
    end
    Core_old=Core;
    cost_old=0.5*sum(G(:).^2)+0.5*lambda*sum(Core(:).^2);
    stop=0;        
    while ~stop
        Core=Core_old-mu_c*Gr;
        if CoreConstr
           Core(Core<0)=0; 
        end
        Rec=reconstructTucker(Core,FACT);
        G=W.*(Rec-X);
        cost=0.5*sum(G(:).^2)+0.5*lambda*sum(Core(:).^2);
        if cost<=cost_old+eps 
            mu_c=mu_c*1.2;
            stop=1;
        else
            mu_c=mu_c/2;
        end
    end
end
