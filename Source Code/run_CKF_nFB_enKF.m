% RUN_CKF_NFBKF - forward-backward algo + linear systems for one of N users
%               - implements (4)-(15) of http://arxiv.org/abs/1110.2098
%
% Syntax: [mu_pri Cov_pri mu_pos Cov_pos mu_tot Cov_tot Cov_lag mu_hat Cov_hat] = run_FBKF_n(param, Y, n)
%
% Inputs:
%    param  - parameters for Kalman filter/smoother
%    Y      - observed measurements
%    param  - struct with Kalman parameters
%               - muX - state mean
%               - CovX - state covariance
%               - CovQ - state noise covariance
%               - CovR - measurement noise covariance
%               - At - state transition matrix (time dependent)
%               - Ht - measurement matrix (time dependent)
%    Y      - observation at each time step
%    n      - user id
%
% Outputs:
%    [mu_pri Cov_pri]   - a priori estimate and covariance
%    [mu_pos Cov_pos]   - a posteriori estimate and covariance
%    [mu_tot Cov_tot]   - smoothed estimate and covariance
%    [mu_hat Cov_hat]   - prediction of x_0 and covariance
%    [Cov_lag]          - lag covariance
%
%
% Author: John Z. Sun
% IBM/MIT
% email: johnsun@mit.edu
% Aug 2011; Last revision: 10-31-2012

%------------- BEGIN CODE --------------

function [mu_pri Cov_pri mu_pos Cov_pos mu_tot Cov_tot Cov_lag mu_hat Cov_hat] ...
                                                    = run_CKF_nFB_enKF(param, Y, n)


    % extract parameters from struct
    K = param.K;
    T = param.T;
    E = param.E;
    muX = param.muE{n};    
    CovX = param.CovX{n};

    
    % initiate output structures
    mu_pri = zeros(K,E,T);
    mu_pos = zeros(K,E,T);
    mu_tot = zeros(K,E,T);
    Cov_pri = zeros(K,K,T);
    Cov_pos = zeros(K,K,T);
    Cov_tot = zeros(K,K,T);
    Cov_lag = zeros(K,K,T);

    Jt = zeros(K,K,T);
    Kt = cell(T,1);
    
    
    % forward (filter) algorithm
    for t = 1:T
   
        A = param.At{t,n};
        H = param.Ht{t,n};
        y = Y{t,n};
        
        CovQ = param.varQ*eye(size(A,1));
        CovR = param.varR*eye(size(H,1));
        % check to see if there are no observations for this step
        noObs = isempty(y);
        
        % predict (time) step
        if t == 1
            % init
            for e=1:E
                mu_pri(:,e,t) = A * muX{e};
            end
            Cov_pri(:,:,t) = cov(mu_pri(:,:,t)');
            % size(Cov_pri(:,:,t))
           
            %Cov_pri(:,:,t) = A*CovX*A' + CovQ;
            Jt(:,:,t) = (CovX * A') / Cov_pri(:,:,t);
        else
            for e=1:E
                mu_pri(:,e,t) = A * mu_pos(:,e,t-1); 
            end
            Cov_pri(:,:,t) = cov(mu_pri(:,:,t)') + CovQ;
            
            % Cov_pri(:,:,t) = A*Cov_pos(:,:,t-1)*A' + CovQ;
            Jt(:,:,t) = (Cov_pos(:,:,t-1) * A') / Cov_pri(:,:,t);
        end
            
        % update (measurement) step
        if noObs
            for e=1:E
                mu_pos(:,e,t) = mu_pri(:,e,t);
            end
            Cov_pos(:,:,t) = Cov_pri(:,:,t);
        else
            Kt{t} = (Cov_pri(:,:,t) * H') / (H*Cov_pri(:,:,t)*H' + CovR);
            for e=1:E
                mu_pos(:,e,t) = mu_pri(:,e,t) +  Kt{t} * (y - H*mu_pri(:,e,t));
            end
            
            Cov_pos(:,:,t) = Cov_pri(:,:,t) - Kt{t} *  H * Cov_pri(:,:,t);
        end
    end
    
    
    % backward (smoother) algorithm
    for t = T:-1:1   
        A = param.At{t,n};
        H = param.Ht{t,n};
        
        if t == T 
            % compute smoothed estimate and covariance
            for e=1:E
                mu_tot(:,e,t) = mu_pos(:,e,t);
            end
            
            Cov_tot(:,:,t) = Cov_pos(:,:,t);
            % compute lag covariance
            if noObs
                Cov_lag(:,:,t) = A * Cov_pos(:,:,t-1);
            else
                Cov_lag(:,:,t) = (eye(K) - Kt{t}*H) * A * Cov_pos(:,:,t-1);
            end
            
        else    
            % compute smoothed estimate and covariance
            for e=1:E
                mu_tot(:,e,t) = mu_pos(:,e,t) + Jt(:,:,t+1) * (mu_tot(:,e,t+1) - mu_pri(:,e,t+1));
            end
            Cov_tot(:,:,t) = Cov_pos(:,:,t) + Jt(:,:,t+1) * (Cov_tot(:,:,t+1) - Cov_pri(:,:,t+1)) * Jt(:,:,t+1)';
            % compute lag variance
            Cov_lag(:,:,t) = Cov_pos(:,:,t) * Jt(:,:,t)' + ...
                    Jt(:,:,t+1)*(Cov_lag(:,:,t+1) - param.At{t+1,n}*Cov_pos(:,:,t))*Jt(:,:,t)';
        end
    end
    
    % find mu_hat and Cov_hat - estimate of X_0
    for e=1:E
        mu_hat{e} = muX{e} + Jt(:,:,1) * (mu_tot(:,e,1) - mu_pri(:,e,1));
    end
    Cov_hat = CovX + Jt(:,:,1) * (Cov_tot(:,:,1) - Cov_pri(:,:,1)) * Jt(:,:,1)';
    
    mu_pri = squeeze(mean(mu_pri,2));
    mu_pos = squeeze(mean(mu_pos,2));
    mu_tot = squeeze(mean(mu_tot,2));
    temp=zeros(K,1);
    for e=1:E
        temp = temp+mu_hat{e};
    end
    mu_hat = temp/E;
   
end