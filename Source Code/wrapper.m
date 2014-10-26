%%% wrapper for learning for CKF -- implements http://arxiv.org/abs/1110.2098
% Procedure is as follows
%   1) Generate model data
%   2) Run CKF on true parameters to set baseline
%   3) Perform learning and CKF estimation concurrently
%   4) Process and display results
%
%
% Author: John Z. Sun
% IBM/MIT
% email: johnsun@mit.edu
% Aug 2011; Last revision: 10-31-2012

%% ------------- BEGIN CODE --------------
    
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s)
%% Loading Gold data
disp('Load data')
if exist('torun','var')
    load(strcat(['data_CKF_test' num2str(torun)]))
    clusterval=torun;
else  
    load data_CKF_test
    clusterval=0;
end
disp('-------------------------------------------------------------------')

%% EM using KF(enKF) and KS(enKS)

% specify struct of Kalman filter / Ensemble Kalman Filter parameters
p_init.K = K;                           % # factors
p_init.M = M;                           % # items
p_init.N = N;                           % # users
p_init.T = T;                           % # time slots
p_init.E = E;                           % # ensemble members
p_init.varU = varU;                     % variance of U 
p_init.varV = varV;                     % variance of H (or V)
p_init.varQ = varQ;                     % variance of process noise
p_init.varR = varR;                     % variance of measurement noise
p_init.A = A;                           % true A matrix
p_init.V = V;                           % true V matrix
p_init.mu = cell(N,1);                  % a priori init state
p_init.CovX = cell(N,1);                % a priori covariance
p_init.At = cell(T,N);                  % time-dependent process transformation
p_init.Ht = cell(T,N);                  % time-dependent measurement
Y = cell(N,T);                          % observations at each time for user n

temp = cell(E);
tempM = zeros(K,1);
for e=1:E
    temp{e} = rand(K,1);
    tempM = temp{e}+tempM;
end
tempM = tempM/E;
for n = 1:N
    p_init.muE{n} = temp;  
    p_init.mu{n} = tempM;
    p_init.CovX{n} = varU*eye(K);       
    for t = 1:T
        p_init.At{t,n} = A;
        p_init.Ht{t,n} = V(obsTensor(:,n,t),:);
        Y{t,n} = Yobserved(obsTensor(:,n,t),n,t);
    end
end

tic;
disp('Run baseline estimation assuming true parameters (A,Q,R,H,varU) known')
[~,~,xKAL_base, xRTS_base] = run_CKF_nUsers(p_init, Y);
disp('Baselines saved')
toc;
%%  if learning, guess parameters
disp('Guess parameters - Initializing')
learnP.CovX = 1;            % learn a priori covariance?                        % <--- specify what parameters to learn
learnP.varQ = 1;            % learn process noise variance?
learnP.varR = 1;            % learn measurement noise variance?
learnP.A = 1;               % learn process?
learnP.V = 1;               % learn item factors?

% if learning, change parameters to these                                       % <--- guess parameters
if learnP.A
    Aest = .7*A + .3*randn(K);
    p_init.A = Aest;
end

if learnP.V
    Vest = .8*V + .2*randn(M,K);
    p_init.V = Vest;
end

for n = 1:N
    if learnP.CovX
        p_init.CovX{n} = .8*eye(K);
    end
    for t = 1:T
        if learnP.A
            p_init.At{t,n} = Aest;
        end
        if learnP.V
            p_init.Ht{t,n} = Vest(obsTensor(:,n,t),:);
        end
    end
end
if learnP.varQ
    p_init.varQ = 0.15;
end
if learnP.varR
    p_init.varR = 0.25;
end

    
%% run Expectation-Maximization learning algorithm and predictor
iter =10;                                                                      %  <--- specify # of iterations
disp('-------------------------------------------------------------------')
tic;
disp('Run KalamnFilter and KalmanSmoother in EM learning algorithm')
[p_vec, loglike, xKAL_learn, xRTS_learn]= run_CKF_EM(p_init, Y, iter, obsTensor, learnP);
toc;
disp(' ')
disp('-------------------------------------------------------------------')
tic;
disp('Run EnsembleKalmanFilter and EnsembleKalmanSmoother in EM learning algorithm')
[p_vec1, loglike1, xKAL_learn_En, xRTS_learn_En] = run_CKF_EM_EnKF(p_init, Y, iter, obsTensor, learnP);
toc;

if exist('torun','var')
    rmsesvd = 0.0;
else
    disp(' ')
    disp('-------------------------------------------------------------------')
    tic;
    disp('Run objective function minimization')
    lambda1 = 0.1;
    lambda2 = 0.1;
    rmsesvd = svdtimein(Ytrue,obsTensor,K,lambda1,lambda2,5);
    toc;
end


%% plot results
plot_Results(p_vec,p_vec1, iter, xKAL_learn, xRTS_learn, xKAL_learn_En, xRTS_learn_En, xKAL_base, xRTS_base,rmsesvd,clusterval);


%% end simulation
disp('End...')