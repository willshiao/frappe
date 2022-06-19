function [FACT,ind]=sortFACT(FACT)
% Sort the factors according to their frobenius norm
T=ones(1,size(FACT{1},2));
for k=1:length(FACT)
    T=T.*sum(FACT{k}.^2);
end
[val,ind]=sort(T,'descend');
for k=1:length(FACT)
    FACT{k}=FACT{k}(:,ind);
end
