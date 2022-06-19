%---------------------------------------------------
%AUTHORS: Sofia Fernandes, Hadi Fanaee-T, Joao Gama
%--------------------------------------------------

function [redundant]=check_redundancy(T,F,search_log,nredundantcomps,delta)
%------------------------------
% INPUT
%   T [double array]: tensor data 
%   F [int]: number of components to consider
%   search_log [int]: vector with the list of visited number of factors, 
%       whose entry i is the number of components visited/considered in the
%       i^th iteration of b-NORMO
%   nredundantcomps [double] vector with the number of redundant components
%       whose entry i is the number pairs of redundant components found in
%       the i^th iteration of b-NORMO
%   delta [double]: correlation threshold (default = 0.7)
%------------------------------
% OUTPUT
%   redundant [bool]: true if there are redundate components adn false,
%       otherwise
%------------------------------
% DESCRIPTION
%   The funcion checks if there are redundant components when decomposing
%   with F components
%------------------------------

if ismember(F,search_log)
    redundant=(nredundantcomps(search_log==F))>0;
else 
    %decompose tensor with F components
    rng('default'); rng(0)
    Tr=parafac(T,F); 

    %compute the correlation matrix
    c=mean_correlation(Tr,F);

    %check if there is redundancy in the decompsoition output
    redundant=(sum(sum(c>delta)>=1))>0;
end