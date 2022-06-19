%---------------------------------------------------
%AUTHORS: Sofia Fernandes, Hadi Fanaee-T, Joao Gama
%--------------------------------------------------
clc
clear all

%load dataset
load('datasets/synth10_50_1');

%convert tensor format to double (if needed)
T=double(T);  

%apply NORMO
R_e=[]; t_e=[];
R_b=[]; t_b=[];
for seed=[0]
    %apply exhaustive search NORMO
    [nredndantcomps,t_e(end+1)]=e_NORMO(T,25,0.7,seed);
    %get number of components for which there were redundancy
    Rs_with_redundancy=find(nredndantcomps>0);
    %get the maximal number of components R such there is no redundancy
    %when decomposing with R or with <R components but there is redundancy
    %when decomposing with R+1 components
    R_e(end+1)=Rs_with_redundancy(1)-1;
    
    %apply binary search NORMO
    [R_b(end+1),t_b(end+1)]=b_NORMO(T,13,0.7,seed);
end
R_e=mode(R_e); avg_t_e=mean(t_e);std_t_e=std(t_e);
R_b=mode(R_b); avg_t_b=mean(t_b);std_t_b=std(t_b);

%output results
fprintf('Approach | Estimate |    Time(s)\n')
fprintf('exh-NORMO  | %8d | %2.2f+/-%2.2f\n',R_e, avg_t_e, std_t_e);
fprintf('bin-NORMO  | %8d | %2.2f+/-%2.2f\n',R_b, avg_t_b, std_t_b);
