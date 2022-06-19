function [A,Tnew]=tmult(T,M,n)
% The Tensor multiplication also named the n-mode multiplication
% turns for instance a 3-way array T(m x p x q) by multiplying M(r x p) 
% along second dimension into A(m x r x q)
%
% Input
% T     tensor
% M     Jxsize(T,n)-matrix
% n     dimension to multiply;
%
% Output:
% A     tensor
% Tnew  matrix given by M*matrizicing(T,n)


Dt=size(T);
Dm=size(M);
Tn=matrizicing(T,n);
Tnew=M*Tn;
Dt(n)=Dm(1);

% bug fix: 21/2-2015
Dt(Dt==0)=1;

A=unmatrizicing(Tnew,n,Dt);

     

    
    