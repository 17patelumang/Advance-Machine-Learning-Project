function [finalrmse]=YtrueYnew()
load data_CKF_test
Ytruechange =Ytrue;
%size(Ytruechange)
load data_CKF_test1

Yttrue=[];
for c=1:nc
Yttrue = [Yttrue,Ytruechange(:,find(as==c),:)];
end

Ynew=[];
for c=1:nc
    load(strcat(['Yval' num2str(c) '.mat']))
    %size(Yhat_rts_learn)
    Ynew = [Ynew, Yhat_rts_learn];
end
finalrmse = myRMSE(Ynew-Yttrue);
end

% get rmse for matrix
function out = myRMSE( in )
    ind = find(isnan(in(:)));
    in(ind)=0;
    out = sqrt(sum(in(:).^2)/length(in(:)));
end