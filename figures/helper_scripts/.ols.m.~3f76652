function [cope,varcope,tstat,F,Fdof]=ols(data,des,tc,Ftests)
% [COPE,VARCOPE,TSTAT,F,Fdof]=ols(DATA,DES,TC,Ftests)
% DATA IS T x V
% DES IS T x EV (design matrix)
% TC IS NCONTRASTS x EV  (contrast matrix)
% Ftests IS CELL ARRAY, NFTESTS*1; each cell contains vector that indexes contrast
%    matrix
%
% see ols_examples.m
%
% TB 2004/LH 2016

if(size(data,1)~=size(des,1))
  error('OLS::DATA and DES have different number of time points');
elseif(size(des,2)~=size(tc,2))
  error('OLS:: DES and TC have different number of evs')
end


pdes=pinv(des);
prevar=diag(tc*pdes*pdes'*tc');
R=eye(size(des,1))-des*pdes;
tR=trace(R);
pe=pdes*data;
cope=tc*pe; 
if(nargout>1)
    res=data-des*pe;
    sigsq=sum(res.*res/tR);
    varcope=prevar*sigsq;
    if(nargout>2)
        tstat=cope./sqrt(varcope);
    end
    
    if (nargout>3)
        %see appendix of Henson and Penny, 2005
        for i = 1:size(Ftests,1)
            corth = eye(size(des,2)) - tc(Ftests{1},:)'*pinv(tc(Ftests{1},:)'); 
                %orthogonal contrast to C
            desorth = des*corth; %design matrix of reduced model
            residR = eye(size(des,1)) - des*pinv(des); %residual forming matrix of full model
            residO = eye(size(des,1)) - desorth*pinv(desorth); %residual forming matrix of reduced model
            projectM = residO-residR; %projection matrix M due to X1
            for j = 1:size(data,2)
                F(i,j) = ((cope(:,j)'*des'*projectM*des*cope(:,j))./(data(:,j)'*residR*data(:,j)))*...
                    (size(des,1)-rank(des))/rank(projectM*des); %F-statistic
            end
            %degrees of freedom for this F-statistic:
            Fdof(i,1) = rank(projectM*des);
            Fdof(i,2) = (size(des,1)-rank(des));
        end
    end
end