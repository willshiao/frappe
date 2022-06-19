%---------------------------------------------------
%AUTHORS: Sofia Fernandes, Hadi Fanaee-T, Joao Gama
%--------------------------------------------------

function [F,t]=b_NORMO(T,F,delta,seed)
%------------------------------
% INPUT
%   T [double array]: tensor data 
%   F [int]: initial number of components to consider
%   delta [double]: correlation threshold (default = 0.7)
%   seed [int]: random seed for reproducibility (default = 0)
%------------------------------
% OUTPUT
%   F [int]: CP model order estimation for data T
%   t [double]: running time (in seconds)
%------------------------------
% DESCRIPTION
%   This function applies NORMO using binary search in the input tensor T,
%   by considering an initial estimate of F components
%------------------------------

%set default threshold as 0.7
if nargin==2
    delta=0.7;
end
if nargin<=3
    seed=0;
end

%set initial parameters
Fmin=2;
Fmax=2*F-1;

ctr=1;

rng('default'); rng(seed)
tic;
while Fmax-Fmin>1
    %keep track of the F's tested
    search_log(ctr)=F;
  
    %decompose tensor with F components
    Tr{ctr}=parafac(T,F);
  
    %compute the correlation matrix
    c{ctr}=mean_correlation(Tr{ctr},F);

  
    %compute number of redundant components for decomposition with F components
    nredundantcomps(ctr)=sum(sum(c{ctr}>delta)>=1);  
    ctr=ctr+1;
  
    %update F estimate
    if nredundantcomps(end)==0
        Fmin=F;
        F=round((F+Fmax)/2);   
    else
        Fmax=F;
        F=round((Fmin+F)/2);    
    end
end

%check if there is a good candidate   
Fmin_visited=ismember(Fmin,search_log);
Fmax_visited=ismember(Fmax,search_log);

if ~Fmin_visited
    redundant=check_redundancy(T,Fmin,search_log,nredundantcomps,delta);

    if ~redundant
        F=Fmin;
    else
        F=1;
    end
elseif ~Fmax_visited
    redundant=check_redundancy(T,Fmax,search_log,nredundantcomps,delta);

    if redundant
        F=Fmin;
    else
        F=Fmax;
    end
else
    F=Fmin;
end

t=toc;







