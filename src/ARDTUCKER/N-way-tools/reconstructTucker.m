%--------------------------------------------------------------------------
% Reconstructs the data from Core and FACT in the TUCKER model
function Rec = reconstructTucker(Core, FACT)
Rec=Core;
for i=1:length(FACT)
    Rec=tmult(Rec,FACT{i},i);
end
