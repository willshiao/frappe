%-------------------------------------------------------------------------- 
function Core=core_init(noc)
       Core=zeros(noc);
       Nx=length(noc);
       if Nx==2
           for k=1:min(noc(1))
               Core(k,k)=1;
           end
       elseif Nx==3
           for k=1:min(noc(1))
               Core(k,k,k)=1;
           end
       elseif Nx==4
           for k=1:min(noc(1))
               Core(k,k,k,k)=1;
           end           
       elseif Nx==5
           for k=1:min(noc(1))
               Core(k,k,k,k,k)=1;
           end           
       elseif Nx==6
           for k=1:min(noc(1))
               Core(k,k,k,k,k,k)=1;
           end           
       else
           disp('Diagonal Core not supported, please initialize core manually')
       end
