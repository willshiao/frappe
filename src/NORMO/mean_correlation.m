%---------------------------------------------------
%AUTHORS: Sofia Fernandes, Hadi Fanaee-T, Joao Gama
%--------------------------------------------------

function c=mean_correlation(Tr,f)
%------------------------------
% INPUT
%   Tr [cell]: CP decomposition
%   f [int]: number of CP factors/components
%------------------------------
% OUTPUT
%   c [double]:  fxf matrix whose entry (i,j) is the average correlation
%       between components i nd j in Tr
%------------------------------
% DESCRIPTION
%   The funcion computes the average correlation (as proposed in the
%   manuscript) for each pair of components in decomposition Tr
%------------------------------

c1 = zeros(f);
c2 = zeros(f);
c3 = zeros(f);
c4 = zeros(f);

for i=1:f
    for j=i+1:f
          c1(i,j)=abs(corr(Tr{1}(:,i),Tr{1}(:,j)));
          c2(i,j)=abs(corr(Tr{2}(:,i),Tr{2}(:,j)));
          c3(i,j)=abs(corr(Tr{3}(:,i),Tr{3}(:,j)));
          if size(Tr) > 3
              c4(i,j)=abs(corr(Tr{4}(:, i), Tr{4}(:, j)));
          end
    end 
end
if size(Tr) > 3
    c=(c1+c2+c3+c4)/4;
else
    c=(c1+c2+c3)/3;
end
