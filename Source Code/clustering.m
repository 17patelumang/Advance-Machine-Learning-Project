% GENERATE_CKF_DATA - create simulated dataset for testing CKF
%
% Inputs:
%   (M,N,K,T)                   - # of items, users, latent factors and time slots respectively
%   (varU, varV, varQ, varR)    - variances of model
%   Oratio                      - percentage of data in training set
%   seed                        - set randomization
%
% Outputs:
%   A                           - stationary transition process
%   Ut                          - user factor tensor
%   V                           - item factor matrix
%   Ytrue                       - true preference tensor
%   Yobserved                   - observed preference tensor
%   obsTensor                   - binary observation tensor
%
%
% Author: Umang Patel and Ilambharathi Kanniah
% Aug 2011; Last revision: 10-31-2012

%------------- BEGIN CODE --------------

function [clusterRMSE]=clustering(max_cluster)

clusterRMSE = zeros(max_cluster,1);
for itercluster=1:1:max_cluster
nc = itercluster;
load data_CKF_test
Ylast = Ytrue(:,:,1);
Ylastobs = Yobserved(:,:,1);
size(find(Ylast(:)));
size(find(Ylastobs(:)));
%moviesmean = mean(Ylastobs,2);
[U X Q]=svd(Ylastobs);

Options.MaxIter = 500;
warning off;
as = kmeans(Q,nc,'Options',Options);

disp(strcat(['Clustering completed. Number of user clusters formed -' num2str(nc)]))
disp('-------------------------------------------------------------------')
for c=1:nc
    load data_CKF_test
    torun=c;
    Ytrue = Ytrue(:,find(as==c),:);
    Yobserved = Yobserved(:,find(as==c),:);
    N = length(find(as==c));
    Ut = Ut(find(as==c),:,:);
    obsTensor = obsTensor(:,find(as==c),:);
    %clear Ylast Ylastobs X U Q ans xKAL_base xRTS_base xKAL_learn xRTS_learn
    clearvars -except A E K M N Options Oratio T Ut V Yobserved Ytrue as c nc obsTensor seed torun varQ varR varU varV clusterRMSE iter itercluster max_cluster
    save(strcat(['data_CKF_test' num2str(c)]))
    torun=c;
    wrapper
end
final = YtrueYnew;
clusterRMSE(itercluster,1)=final;
end
figure
hbars = bar(clusterRMSE,0.4);
set(hbars(1),'BaseValue',1);
xlabel('Number of Cluster(s)')
ylabel('Root Mean Square Error (RMSE)')
title('Clustering of Users and Running the Model')
end


