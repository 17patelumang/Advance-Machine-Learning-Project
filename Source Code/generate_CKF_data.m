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
% Author: John Z. Sun
% IBM/MIT
% email: johnsun@mit.edu
% Aug 2011; Last revision: 10-31-2012

%------------- BEGIN CODE --------------


function generate_CKF_data

    % specify dataset parameters
    M = 2000;             % # items
    N = 3000;             % # users
    K = 10;               % # factors
    T = 5;              % # time slots
    E = 100;
    varU = 0.1;            % true variance of U 
    varV = 1;            % true variance of V
    varQ = 0.05;         % true variance of process noise
    varR = 0.1;          % true variance of measurement noise
    Oratio = 0.2;        % size of training set
    seed = 100;          % random seed used
    
    % specify dataset variables
    Ut = zeros(N,K,T);                  % user factor tensor
    Ustart = sqrt(varU)*randn(N,K);     % initial user factor matrix
    V = sqrt(varV)*randn(M,K);          % item factor matrix
    Ytrue = zeros(M,N,T);               % true time-dependent preference tensor
    Yobserved = zeros(M,N,T);           % observed preference tensor
    obsTensor = false(M,N,T);           % binary observation tensor
    
    % specify stationary process transformation matrix                          <-- can specify this
    A = eye(K) + .3*[zeros(K,1) [eye(K-1); zeros(1,K-1)]] - .3*[zeros(1,K); [eye(K-1) zeros(K-1,1)]];
    for k = 1:K
        A(k,:) = A(k,:)/norm(A(k,:))*((varU - varQ)/varU);
    end
    
    
    % run dynamics
    for t = 1:T
        % update state - with process noise
        if t == 1
            Ut(:,:,t) = (A*Ustart' + sqrt(varQ)*randn(K,N))';
        else
            Ut(:,:,t) = (A*Ut(:,:,t-1)' + sqrt(varQ)*randn(K,N))';
        end
        % make observations - with measurement noise
        Ytrue(:,:,t) = V*Ut(:,:,t)' + sqrt(varR)*randn(M,N);
    end

    
    % randomly generate observations
    numTrain = round(Oratio*M*N*T);  
    % find observed values - make sure training set contains each user and item at least once
    isTrainOK = 0;
    while ~isTrainOK
        B = randperm(M*N*T);
        Otemp = zeros(M,N,T);
        Otemp(B(1:numTrain)) = 1;
        isTrainOK = (min(sum(sum(Otemp,3),1)) * min(sum(sum(Otemp,3),2))) > 0;
        if ~isTrainOK
            disp('Problem! Need at least one observation per user and item. Fixing now...');
        end
    end
    
    
    % Code for Extracting train and probe vector for Bayesian PMF
    % extension.
    % fill observation tensor
    obsTensor(B(1:numTrain)) = true;
    Yobserved(B(1:numTrain)) = Ytrue(B(1:numTrain));
    % size(B(1:numTrain))
    % max(max(max(Ytrue)))
    % min(min(min(Ytrue)))
    %Ysq = squeeze(Yobserved(:,:,T));
    % length(find(Ysq(:)))
    %     train_vec = zeros(length(find(Ysq(:))),3);
    %     probe_vec = zeros(N*M,3);
    %     traint = 1;
    %     probet = 1;
    %     for n=1:N
    %         for m=1:M
    %             if Ysq(m,n)~=0
    %                 train_vec(traint,1)=n;
    %                 train_vec(traint,2)=m;
    %                 train_vec(traint,3)=Ysq(m,n);
    %                 traint=traint+1;
    %             end
    %             probe_vec(probet,1)=n;
    %             probe_vec(probet,2)=m;
    %             probe_vec(probet,3)=Yobserved(m,n,T);
    %             probet=probet+1;
    %         end
    %     end    
    clear Otemp Ustart isTrainOK t k numTrain B 
    save data_CKF_test
    %save('moviedata.mat','train_vec','probe_vec');      
end

