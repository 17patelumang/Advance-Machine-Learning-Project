function [U,V,RMSE]=stograd_func(Otrue,Oobv,K,lambda1,lambda2,maxiter)
if nargin<6
    maxiter = 10;
end

% set up data
N = size(Otrue,1);   % number of measurements
M = size(Otrue,2);   % number of variables

U = rand(N,K);
V = rand(M,K);

%% CVX Code
myOBJ=@(Otrue,Oobv,U,V) CVX_min(Otrue,Oobv,U,V);

for i=1:maxiter
cvx_begin quiet
    variable U(N,K)
    minimize(myOBJ(Otrue,Oobv,U,V)+lambda1*(square_pos(norm(U,'fro')))+lambda2*(square_pos(norm(V,'fro'))))
cvx_end

cvx_begin quiet
    variable V(M,K)
    minimize(myOBJ(Otrue,Oobv,U,V)+lambda1*(square_pos(norm(U,'fro')))+lambda2*(square_pos(norm(V,'fro'))))
cvx_end

%disp(['Learning iteration ' int2str(i) ]);
end

RMSE = myRMSE((U*V')-Otrue);
end
% fprintf('\nObjective function value from CVX is :- %s \n', num2str(cvx_value));

function [value]=CVX_min(Otrue,Oobv,U,V)
    vals = Oobv.*(Otrue-U*V');
    value = sum(square(vals(:)));
end

function out = myRMSE( in )
    out = sqrt(sum(in(:).^2)/length(in(:)));
end
