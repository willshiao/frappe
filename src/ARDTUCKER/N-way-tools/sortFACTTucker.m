function [Core,FACT,P]=sortFACTTucker(Core,FACT)
% Sort the factors according to their frobenius norm
for k=1:length(FACT)
    T=sum(FACT{k}.^2);
    [val,ind]=sort(T,'descend');
    P{k}=sparse(1:length(ind),ind,ones(1,length(ind)));
    FACT{k}=FACT{k}*P{k};
    Core=tmult(Core,P{k}',k);
end

