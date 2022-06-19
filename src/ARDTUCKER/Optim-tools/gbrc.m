%----------------------------------------------------------
function [A,mu,cost_ls,cost_reg]=gbrc(XSt,SSt,Aold,mu,lambda,constr,iter)
%   Gradient Based ridge Coding
%       solves
%           argmin_A 0.5||X-AS||_F^2+0.5lambda||A||_F
%
%   Usage:

%
%   Input
%           XSt             X*S'
%           SSt             S*S'
%           Aold            Initial value of A
%           mu              initial step size
%           lambda          L1-regularization strength
%           constr          1: non-negativity, 0: unconstrained optimization
%
%   Output
%           A               solution matrix
%           mu              step size
%           cost_LS         0.5*(||X-AS||_F^2-||X||_F^2)
%           cost_reg        0.5lambda||A||_F

tol=1e-9;
dA=inf;
if length(lambda)<size(SSt,1)
   lambda=repmat(lambda,1,size(SSt,1)); 
end
cost=-sum(sum(XSt.*Aold))+0.5*sum(sum((Aold'*Aold).*SSt))+0.5*sum(Aold.^2*lambda');
N=size(Aold,1);
h_inv=1./(diag(SSt)+lambda')';        
k=1;
while k<iter && dA>tol
    cost_old=cost;
    grad=Aold*SSt-XSt+repmat(lambda,size(Aold,1),1).*Aold;
    grad=grad.*repmat(h_inv,N,1);            
    stop=0;
    while ~stop 
      A=Aold-mu*grad;
      if constr % Non-negativitity
           A(A<0)=0;
      end
      cost_ls=-sum(sum(XSt.*A))+0.5*sum(sum((A'*A).*SSt));
      cost_reg=0.5*sum(A.^2*lambda');
      cost=cost_ls+cost_reg;
      if cost_old>=cost
           mu=1.2*mu;
           stop=1;
           dA=sum(sum((Aold-A).^2))/(sum(sum(A.^2))+eps);
           Aold=A;
      else
           mu=mu/2;
      end
    end
end

