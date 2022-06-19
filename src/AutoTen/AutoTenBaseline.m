function [Fac, F_est] = AutoTenBaseline(X,Fmax,strategy)
%Vagelis Papalexakis - Carnegie Mellon University, School of Computer
%Science (2015-2016)

%strategy--> 1: Frobenius norm PARAFAC with non-negativity consraints
%strategy--->2: KL-Divergence PARAFAC

allF = 2:Fmax;
all_Fac_fro = {};
all_Fac_kl = {};
normX = norm(X);
all_loss = zeros(length(allF),1);
SMALLNUMBER = 10^-6;

for f = allF
    
    curr_idx = find(f == allF);
   if (strategy==1)%frobenius
       Fac = cp_als(X,f,'tol',10^-6,'maxiters',10^2);
       all_loss(curr_idx) = sqrt( normX^2 + norm(Fac)^2 - 2 * innerprod(X,Fac) );
   else %KL
       Fac = cp_apr(X,f);
       all_loss(curr_idx) = tt_loglikelihood(X,Fac);
   end
   %termination criterion
   if(curr_idx>1)
      if( abs(all_loss(curr_idx) - all_loss(curr_idx-1))/all_loss(curr_idx-1)<=SMALLNUMBER)
         Fac = oldFac;
         break;
      end
   end
   oldFac = Fac;
end

F_est = size(Fac.U{1},2);