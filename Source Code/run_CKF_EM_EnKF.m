% RUN_CKF_EM - EM learning algorithm for CKF
%               - implements (16)-(20) of http://arxiv.org/abs/1110.2098
%
% Syntax: [p_vec loglike xKal xRTS] = run_CKF_EM(p_init, Y, iter, O, learnP)
%
% Inputs:
%    p_init     - initial parameters for Kalman filter/smoother
%    Y          - observations in time per user
%    O          - binary matrix of actual observations
%    learnP     - indicators for which parameters to learn%
%
% Outputs:
%    p_vec -        array of parameters for each iteration
%    loglike -      log-likelihood of innovations for each iteration
%    xKAL_learn -   Kalman filter output for each iteration
%    xRTS_learn -   Kalman smoother output for each iteration
%
%
% Author: John Z. Sun
% IBM/MIT
% email: johnsun@mit.edu
% Aug 2011; Last revision: 10-31-2012

%------------- BEGIN CODE --------------

function [p_vec loglike xKAL_learn xRTS_learn] = run_CKF_EM_EnKF(p_init, Y, iter, obsTensor, learnP)

    % create structure to hold EM
    p_vec = p_init;
    p_vec(iter+1) = p_init;
    loglike = zeros(iter,1);
    xKAL_learn = cell(iter,1);
    xRTS_learn = cell(iter,1);
    
    
    % run iterations
    for i = 2:iter+1
        
        % important params for convenience
        At = p_vec(i-1).At;
        Ht = p_vec(i-1).Ht;
        T = p_vec(i-1).T;
        K = p_vec(i-1).K;
        M = p_vec(i-1).M;
        N = p_vec(i-1).N;
        
        
        % run Kalman filter for each user - E step
        [x_at0, ~, x_pos, x_tot, P_out, ~, ~, P_tot, P_lag] = run_CKF_nUsers_EM(p_vec(i-1), Y);
%         [x_at0, temp1, x_pos, x_tot, P_out, temp2, temp3, P_tot, P_lag] = run_CKF_nUsers(p_vec(i-1), Y);       % <-- deprecated notation
        xKAL_learn{i-1} = zeros(N,K,T);
        xRTS_learn{i-1} = zeros(N,K,T);
        for n = 1:N
            xKAL_learn{i-1}(n,:,:) = reshape(x_pos{n}, 1, K, T);
            xRTS_learn{i-1}(n,:,:) = reshape(x_tot{n}, 1, K, T);
        end
        
        
        % helper functions
        fAQ1 = 0;
        fAQ2 = 0;
        fAQ3 = 0;
        fR1 = 0;
        fR2 = 0;
        fR3 = 0;
        fH1 = zeros(M,K);
        fH2 = cell(N,1);
        fU = 0;
        for n = 1:N
            % give entire set of smoothed estimates
            x_temp = [x_at0{n} x_tot{n}];
            P_temp = cat(3,P_out{n}, P_tot{n});
            % estimate varU
            fU = fU + trace(P_out{n} + x_at0{n}*x_at0{n}');
            H_temp = zeros(K,K,T);
            for t = 1:T
                % for A and Q
                fAQ1 = fAQ1 + P_temp(:,:,t) + x_temp(:,t)*x_temp(:,t)';
                fAQ2 = fAQ2 + x_temp(:,t+1)*x_temp(:,t)' + P_lag{n}(:,:,t);
                fAQ3 = fAQ3 + P_temp(:,:,t+1) + x_temp(:,t+1)*x_temp(:,t+1)';
                % for R
                fR1 = fR1 + trace(Y{t,n}*Y{t,n}');
                fR2 = fR2 + trace(Y{t,n}*x_tot{n}(:,t)'*Ht{t,n}');
                fR3 = fR3 + trace(Ht{t,n}*(P_temp(:,:,t+1) + x_temp(:,t+1)*x_temp(:,t+1)')*Ht{t,n}');
                % for H
                Ytemp = zeros(M,1);
                Ytemp(obsTensor(:,n,t)>0) = Y{t,n};
                fH1 = fH1 + Ytemp * x_tot{n}(:,t)';
                H_temp(:,:,t) = x_tot{n}(:,t)*x_tot{n}(:,t)';
            end
            fH2{n} = P_tot{n} +  + H_temp;
            
        end
        
        % finally...
        varU = fU/K/N;
        fQ = fAQ3 - fAQ2*At{1}' - At{1}*fAQ2' + At{1}*fAQ1*At{1}';
        fR = fR1 - 2*fR2 + fR3;
        
        
        % find log likelihood
        loglike(i-1) = -sum(obsTensor(:))/2*log(2*pi);
        for n = 1:N
            for t = 1:T
                Sig = Ht{t,n}*P_tot{n}(:,:,t)*Ht{t,n}' + p_vec(i-1).varR * eye(length(Y{t,n}));
                Mu = Y{t,n} - Ht{t,n} * x_tot{n}(:,t);
                loglike(i-1) = loglike(i-1) - log(det(Sig))/2 - Mu'* (Sig \ Mu)/2;
            end
        end
        disp(['Learning iteration ' int2str(i-1) ]);
        
        
        % optimize parameters - M step
        p_vec(i) = p_vec(i-1);
        
        
        % update varU
        if learnP.CovX
            for n = 1:N
                p_vec(i).CovX{n} = varU * eye(K);
            end
        end
        
        
        % get varQ
        if learnP.varQ
            p_vec(i).varQ = trace(fQ)/K/N/T;
        end
            
        
        % get varR
        if learnP.varR
            p_vec(i).varR = fR/sum(obsTensor(:));
        end
        
        
        % get A
        if learnP.A
            Aout = (fAQ2) / fAQ1;
            p_vec(i).A = Aout;
            for t = 1:T
                for n = 1:N
                    p_vec(i).At{t,n} = Aout;
                end
            end
        end
        
        
        % get H
        Oh = zeros(T,M,N) > 0;
        for t = 1:T
            for n = 1:N
                Oh(t,:,n) = obsTensor(:,n,t)';
            end
        end
        if learnP.V
            Hout = zeros(M,K);
            for m = 1:M
                temp = zeros(K,K,N);
                for n = 1:N
                    temp(:,:,n) = sum(fH2{n}(:,:,Oh(:,m,n)),3);
                end
                Hout(m,:) = fH1(m,:) / sum(temp,3);%(temp + p_vec(i).varR/p_vec(i).varV/2*eye(K));
            end
            p_vec(i).V = Hout;
            for t = 1:T
                for n = 1:N
                    p_vec(i).Ht{t,n} = Hout(obsTensor(:,n,t)>0,:);
                end
            end   
        end
    end
end

%------------- END OF CODE --------------