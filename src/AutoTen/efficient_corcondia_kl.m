function [c,time] = efficient_corcondia_kl(X,Fac)
%Vagelis Papalexakis - Carnegie Mellon University, School of Computer
%Science (2014-2015)

s = size(X);
I = s(1); J = s(2); K = s(3);

C = Fac.U{3}; B = Fac.U{2}; A = Fac.U{1};
A = A*diag(Fac.lambda);
F = size(A,2);
tic

Z2 = reshape(X,[I*J*K 1]);%Z2 is x
disp('Computed Z2')
vecG = regression_kl_efficient(Z2,A,B,C);
disp('Computed vecG')
G = sptensor(reshape(sptensor(vecG), [F F F]));

disp('Computed G')
T = sptensor([F F F]);
for i = 1:F; T(i,i,i) =1; end

c = 100* (1 - sum(sum(sum(double(G-T).^2)))/F);
time = toc;
end

function x = regression_kl_efficient(y,A,B,C)

im_iter = 30; %iterations for the iterative majorization

x = tenrand([size(A,2) size(B,2) size(C,2)]);
if (size(A,2) == 1) %rank one is more expensive because kron_mat_vec requires in that case the "full" version
    y_approx = kron_mat_vec({full(A) full(B) full(C)},x);
else
    y_approx = kron_mat_vec({A B C},x);
end
y_approx = sparse(double(reshape(y_approx,size(y))));
x = reshape(x, [size(A,2)*size(B,2)*size(C,2) 1]);

% norm_const = kron_mat_vec({A' B' C'},reshape(tensor(ones(size(y))), [size(A,1) size(B,1) size(C,1)] ));
% norm_const = double(reshape(norm_const,[1 size(A,2)*size(B,2)*size(C,2)]));
norm_const = kron(kron(sum(A,1),sum(B,1)),sum(C,1));
normalization = repmat(norm_const',1,size(y,2));

for it = 1:im_iter
    part1 = sptensor(double(y)./double(y_approx + eps));
    
    if(size(A,2) == 1) %rank one is more expensive because kron_mat_vec requires in that case the "full" version
        part2 = kron_mat_vec({full(A') full(B') full(C')},reshape(part1,[size(A,1) size(B,1) size(C,1)]));%kron(A,B,C)' * part1
    else
        part2 = kron_mat_vec({A' B' C'},reshape(part1,[size(A,1) size(B,1) size(C,1)]));
    end
    
%     part2 = tensor(reshape(part2,size(x)));%this might not be sparse actually
    part2 = reshape(part2,size(x));%this might not be sparse actually
    
    x = x .* part2;
    
    
    if(size(A,2) == 1)%rank one is more expensive because kron_mat_vec requires in that case the "full" version
        y_approx = kron_mat_vec({full(A) full(B) full(C)},reshape(x,[size(A,2) size(B,2) size(C,2)]));
    else
        y_approx = kron_mat_vec({A B C},reshape(x,[size(A,2) size(B,2) size(C,2)]));    
    end
    y_approx = sptensor(reshape(y_approx,size(y)));%this is usually indeed sparse
    
    disp(sprintf('Iterative Majorization iter %d',it))
end
x = x./normalization;
end

function C = kron_mat_vec(Alist,X)
K = length(Alist);

for k = K:-1:1
    A = Alist{k};
    Y = ttm(X,A,k);
    X = Y;
    X = permute(X,[3 2 1]);
end
C = Y;
end