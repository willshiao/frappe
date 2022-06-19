%----------------------------------------------------------
function [A,mu,cost_ls,cost_reg]=gbsc(XSt,SSt,Aold,mu,lambda,constr,iter)
%   Gradient Based Sparse Coding
%       solves
%           argmin_A 0.5||X-AS||_F^2+lambda||A||_1
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
%           iter            maximal number of iterations
%
%   Output
%           A               solution matrix
%           mu              step size
%           cost_ls         0.5*(||X-AS||_F^2-||X||_F^2)
%           cost_reg        lambda||A||_1

tol=1e-9;
dA=inf;
if length(lambda)<size(SSt,1)
   lambda=repmat(lambda,1,size(SSt,1)); 
end
h_inv=1./diag(SSt+eps)';   
cost=-sum(sum(XSt.*Aold))+0.5*sum(sum((Aold'*Aold).*SSt))+sum(abs(Aold)*lambda');
N=size(Aold,1);
k=1;
while k<iter && dA>tol
    cost_old=cost;
    grad=Aold*SSt-XSt;     
    grad=grad.*repmat(h_inv,N,1);            
    stop=0;
    while ~stop 
      A=Aold-mu*grad;
      T=mu*repmat(lambda.*h_inv,N,1);
      A(abs(A)<T)=0;
      A=A-T.*sign(A);    
      if constr % Non-negativitity
           A(A<0)=0;
      end
      cost_ls=-sum(sum(XSt.*A))+0.5*sum(sum((A'*A).*SSt));
      cost_reg=sum(abs(A)*lambda');
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
    k=k+1;
end
