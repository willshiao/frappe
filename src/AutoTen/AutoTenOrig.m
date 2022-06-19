function [Fac, c, F_est,loss] = AutoTenOrig(X,Fmax,strategy)
%Vagelis Papalexakis - Carnegie Mellon University, School of Computer
%Science (2015-2016)


%strategy = 1--> choose the loss that gives maximum c, among the "best"
%points
%strategy = 2--> choose the loss that gives maximum F, among the "best"
%points

allF = 2:Fmax;
all_Fac_fro = {};
all_Fac_kl = {};
thresh = 20;

all_c_fro = zeros(length(allF),1);
all_c_kl = zeros(length(allF),1);

for f = allF
   Fac_fro = cp_als(X,f,'tol',10^-6,'maxiters',10^2);
   c_fro = efficient_corcondia(X,Fac_fro);
   all_c_fro(find(f == allF)) = c_fro;
   all_Fac_fro{find(f == allF)} = Fac_fro;
   
   Fac_kl = cp_apr(X,f);
   c_kl = efficient_corcondia_kl(X,Fac_kl);
   all_c_kl(find(f == allF)) = c_kl;
   all_Fac_kl{find(f == allF)} = Fac_kl;
end

all_c_fro(all_c_fro<thresh) = 0;
all_c_kl(all_c_kl<thresh) = 0;

% figure;bar(allF,all_c_fro);
% figure;bar(allF,all_c_kl);

[F_fro, c_fro] = multi_objective_optim(allF,all_c_fro);
[F_kl, c_kl] = multi_objective_optim(allF,all_c_kl);

%force change the strategy, if either one of the "c" is zero
if(c_fro == 0 || c_kl == 0)
    strategy = 1;
end

if(strategy == 1)
    [c, ~] = max([c_fro c_kl]);%%%%%
    [~,max_idx] = max([sum(all_c_fro) sum(all_c_kl)]);%this gives us confidence on which one has better estimates
else
    [F_est, max_idx] = max([F_fro F_kl]);
end
if(max_idx == 1)%Fro is better
    Fac = all_Fac_fro{find(F_fro == allF)};
    best_c = all_c_fro;
    loss = 'fro';
else %KL is better
    Fac = all_Fac_kl{find(F_kl == allF)};
    best_c = all_c_kl;
    loss = 'kl';
end

if(strategy == 1)
    F_est = size(Fac.U{1},2);
else
   c =  best_c(find(F_est == allF));
end


end


function [x_best, y_best] = multi_objective_optim(x,y)
    %clustering heuristic method
    try
        clusters = kmeans(y,2,'Distance','cityblock');
    catch %if an error is thrown, then an empty cluster is formed, so then we just assign all elements to cluster 1
        disp('Empty cluster!!')
        clusters = ones(size(y));
        clusters(y>0)=2;   
    end
    cent1 = mean(y(clusters == 1));
    cent2 = mean(y(clusters == 2));
    [maxval, cent_idx] = max([cent1 cent2]);
    x_to_choose = x(clusters == cent_idx);
    [x_best,x_idx] = max(x_to_choose);
    y_best = y(x == x_best);
end
