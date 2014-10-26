% SVD_TIME_IN - Objective Minimization Algorithm for given sparse
% observation matrix
%
% Inputs:
%   Ytrue                       - True observations
%   obsTensor                   - Observation Tensor
%   K                           - Size of the Latent Factor
%   lambda1,lambda2             - Regularization Parameters
%
% Outputs:
%   RMSE                        - Final RMSE values.
%
% Authors : Umang Patel and Ilambharathi Kanniah
% April 2014; Last revision: 05/01/2014

%------------- BEGIN CODE --------------


function [rmse]=svdtimein(Ytrue,obsTensor,K,lambda1,lambda2,maxiter)
if nargin<6
    maxiter = 10;
end

Yest = zeros(size(Ytrue));
for i=1:size(Ytrue,3)
    disp(['Objective Minimization -  Time  ' int2str(i) ]);
    [U,V,~]=stograd_func(Ytrue(:,:,i),obsTensor(:,:,i),K,lambda1,lambda2,maxiter);
    Yest(:,:,i) = U*V';
end

rmse = myRMSE(Yest(:)-Ytrue(:));

end

function out = myRMSE( in )
    out = sqrt(sum(in(:).^2)/length(in(:)));
end


%------------- END CODE --------------