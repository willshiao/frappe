%---------------------------------------------------
%AUTHORS: Sofia Fernandes, Hadi Fanaee-T, Joao Gama
%--------------------------------------------------

function [nredundantcomps,t,Tr,c]=e_NORMO(T,Fmax,delta,seed)
%------------------------------
% INPUT
%   T [double array]: tensor data 
%   Fmax [int]: maximum number of components to consider
%   delta [double]: correlation threshold (default = 0.7)
%   seed [int]: random seed for reproducibility (default = 0)
%------------------------------
% OUTPUT
%   nredundantcomps [double vector]: vector of length Fmax whose entry i is
%       the number of redundant pairs of components found when decomposing 
%       with i components
%   t [double]: running time (in seconds)
%   Tr [cell]: cell with Fmax elements whose entry i is the decomposition
%       output when decomposing with i components
%   c [cell]:  cell with Fmax elements whose entry i is the average
%       correlation  matrix of pairs of components obtained when
%       decomposing with i components
%------------------------------
% DESCRIPTION
%   This function applies NORMO using exhaustive search in the input tensor T,
%   by considering a maximum of Fmax components
%------------------------------

%set default threshold as 0.7
if nargin==2
    delta=0.7;
end
if nargin<=3
    seed=0;
end

rng('default'); rng(seed)
tic;
Options(5) = NaN;
for f=2:Fmax
    %decompose tensor with F components
    Tr{f}=parafac(T,f, Options); 

	%compute the correlation matrix
	c{f}=mean_correlation(Tr{f},f);

    %compute number of redundant components for decomposition with of components
    nredundantcomps(f)=sum(sum(c{f}>delta)>=1);   
end
t=toc;
end
