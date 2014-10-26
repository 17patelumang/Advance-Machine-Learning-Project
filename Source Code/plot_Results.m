% PLOT_RESULTS - plot results comparing EM prediction versus baseline
%
% Syntax: plot_Results(p_vec, iter, xKAL_learn, xRTS_learn, xKAL_base, xRTS_base)
%
% Inputs:
%    p_vec          - array of parameters for each iteration
%    iter           - # runs of learning
%    xKAL_learn     - Kalman filter output for each iteration
%    xRTS_learn     - Kalman smoother output for each iteration
%    xKAL_base      - Kalman filter baseline 
%    xRTS_base      - Kalman smoother baseline
%
% Outputs:
%    Figures
%
%
% Author: John Z. Sun
% IBM/MIT
% email: johnsun@mit.edu
% Aug 2011; Last revision: 10-31-2012
% Edited for Ensemble Kalman Fiter and Smoothers
% Edit by Umang Patel, Ilambharathi Kanniah

%------------- BEGIN CODE --------------


function plot_Results(p_vec, p_vec1, iter, xKAL_learn, xRTS_learn, xKAL_learn_En, xRTS_learn_En, xKAL_base, xRTS_base,rmsesvd,cv)

    %close all
    %figure
    % find true data to compare to
    if cv~=0
        load(strcat(['data_CKF_test' num2str(cv)]))
        disp('Reading..');
    else  
        load data_CKF_test
    end
    

    % storage of learned parameters
    rmseVarU_learn = zeros(iter+1,1);
    rmseVarQ_learn = zeros(iter+1,1);
    rmseVarR_learn = zeros(iter+1,1);
    rmseA_learn = zeros(iter+1,1);
    rmseH_learn = zeros(iter+1,1);
    for i = 1:iter+1
        rmseVarU_learn(i) = myRMSE(varU - p_vec(i).CovX{1}(1));
        rmseVarQ_learn(i) = myRMSE(varQ - p_vec(i).varQ);
        rmseVarR_learn(i) = myRMSE(varR - p_vec(i).varR);
        rmseA_learn(i) = myRMSE(A - p_vec(i).A);
        rmseH_learn(i) = myRMSE(V - p_vec(i).V);
    end
    

    % get baseline results
    U_kal_base = zeros(N,K,T);
    U_rts_base = zeros(N,K,T);
    Yhat_kal_base = zeros(M,N,T);
    Yhat_rts_base = zeros(M,N,T);
    
    % get user factors from baseline Kalman estimates
    for n = 1:N
        U_kal_base(n,:,:) = reshape(xKAL_base{n}, 1, K, T);
        U_rts_base(n,:,:) = reshape(xRTS_base{n}, 1, K, T);
    end
    for t = 1:T
        % predict observation matrix
        Yhat_kal_base(:,:,t) = V * U_kal_base(:,:,t)';
        Yhat_rts_base(:,:,t) = V * U_rts_base(:,:,t)';
    end
    % find RMSE of user factors
    rmseU_kal_base = myRMSE(U_kal_base - Ut);
    rmseU_rts_base = myRMSE(U_rts_base - Ut);
    % find RMSE of predictions
    rmseY_kal_base = myRMSE(Yhat_kal_base - Ytrue);
    rmseY_rts_base = myRMSE(Yhat_rts_base - Ytrue);
    % set baseline of a priori prediction
    
    
    % get learned results
    rmseU_kal_learn = zeros(iter,1);
    rmseU_rts_learn = zeros(iter,1);
    rmseY_kal_learn = zeros(iter,1);
    rmseY_rts_learn = zeros(iter,1);
    
    for i = 1:iter
        % learned user factors
        U_kal_learn = xKAL_learn{i};
        U_rts_learn = xRTS_learn{i};
        Yhat_kal_learn = zeros(M,N,T);
        Yhat_rts_learn = zeros(M,N,T);
        for t = 1:T
            % predict observation matrix
            Yhat_kal_learn(:,:,t) = p_vec(i).V * U_kal_learn(:,:,t)';
            Yhat_rts_learn(:,:,t) = p_vec(i).V * U_rts_learn(:,:,t)';
        end 
        % find RMSE of user factors
        rmseU_kal_learn(i) = myRMSE(U_kal_learn - Ut);
        rmseU_rts_learn(i) = myRMSE(U_rts_learn - Ut);
        % find RMSE of predictions
        rmseY_kal_learn(i) = myRMSE(Yhat_kal_learn - Ytrue);
        rmseY_rts_learn(i) = myRMSE(Yhat_rts_learn - Ytrue);
    end

    
    % ENSEMBLE RMSE VALUES
    % storage of learned parameters
    rmseVarU_learnEn = zeros(iter+1,1);
    rmseVarQ_learnEn = zeros(iter+1,1);
    rmseVarR_learnEn = zeros(iter+1,1);
    rmseA_learnEn = zeros(iter+1,1);
    rmseH_learnEn = zeros(iter+1,1);
    for i = 1:iter+1
        rmseVarU_learnEn(i) = myRMSE(varU - p_vec1(i).CovX{1}(1));
        rmseVarQ_learnEn(i) = myRMSE(varQ - p_vec1(i).varQ);
        rmseVarR_learnEn(i) = myRMSE(varR - p_vec1(i).varR);
        rmseA_learnEn(i) = myRMSE(A - p_vec1(i).A);
        rmseH_learnEn(i) = myRMSE(V - p_vec1(i).V);
    end   
    
    % get learned results
    rmseU_kal_learnEn = zeros(iter,1);
    rmseU_rts_learnEn = zeros(iter,1);
    rmseY_kal_learnEn = zeros(iter,1);
    rmseY_rts_learnEn = zeros(iter,1);
    
    for i = 1:iter
        % learned user factors
        U_kal_learnEn = xKAL_learn_En{i};
        U_rts_learnEn = xRTS_learn_En{i};
        Yhat_kal_learnEn = zeros(M,N,T);
        Yhat_rts_learnEn = zeros(M,N,T);
        for t = 1:T
            % predict observation matrix
            Yhat_kal_learnEn(:,:,t) = p_vec1(i).V * U_kal_learnEn(:,:,t)';
            Yhat_rts_learnEn(:,:,t) = p_vec1(i).V * U_rts_learnEn(:,:,t)';
        end 
        % find RMSE of user factors
        rmseU_kal_learnEn(i) = myRMSE(U_kal_learnEn - Ut);
        rmseU_rts_learnEn(i) = myRMSE(U_rts_learnEn - Ut);
        % find RMSE of predictions
        rmseY_kal_learnEn(i) = myRMSE(Yhat_kal_learnEn - Ytrue);
        rmseY_rts_learnEn(i) = myRMSE(Yhat_rts_learnEn - Ytrue);
    end
    
    % END ENSEMBLE VALUES
    
    if cv~=0
        save(strcat(['Yval' num2str(cv)]))
    else  
        save Yval
    end
    
    
    if cv==0
    disp('Plot results')
    % plot specifications
    disp('Start plotting')
    lineWidth = 2;
    fontsize = 15;
    ColorSet = [0,0,0; ...
                125,0,125; ...
                0,255,255; ...
                125,180,0; ...
                255,150,50;...
                100,255,100;...
                0,100,255;] / 255;
            
    
    % plot parameter prediction
    figure('Position', [100 100 600 500], 'Color', 'w');
    hold all
    plot(0:iter, rmseA_learn,'LineWidth',lineWidth, 'Color', ColorSet(4,:))
    plot(0:iter, rmseH_learn,'LineWidth',lineWidth, 'Color', ColorSet(5,:))
    plot(0:iter, rmseVarQ_learn, 'LineWidth',lineWidth, 'Color', ColorSet(2,:))
    plot(0:iter, rmseVarR_learn, 'LineWidth',lineWidth,'Color', ColorSet(3,:))
    plot(0:iter, rmseVarU_learn, 'LineWidth',lineWidth, 'Color', ColorSet(1,:))
 
  
    legend('A', 'H (subset of V)', 'variance of Q', 'variance of R', 'variance of U',  'Location', 'best')
    xlabel('Iteration')
    ylabel('Root Mean Square Error (RMSE)')
    title('Parameter Prediction ')    
    
    % plot user factor prediction
    figure('Position', [100 100 600 500], 'Color', 'w');
    hold all
    plot(1:iter, rmseU_kal_base*ones(iter,1), '-', 'LineWidth',lineWidth, 'Color', ColorSet(1,:))
    plot(1:iter, rmseU_rts_base*ones(iter,1), '-', 'LineWidth',lineWidth, 'Color', ColorSet(2,:))
    plot(1:iter, rmseU_kal_learn, '-','LineWidth',lineWidth,  'Color', ColorSet(3,:))
    plot(1:iter, rmseU_rts_learn, '-', 'LineWidth',lineWidth, 'Color', ColorSet(4,:))    
    plot(1:iter, rmseU_kal_learnEn, '-','LineWidth',lineWidth, 'Color', ColorSet(5,:))
    plot(1:iter, rmseU_rts_learnEn, '-', 'LineWidth',lineWidth, 'Color', ColorSet(6,:))    
    
    xlabel('Iteration')
    ylabel('Root Mean Square Error (RMSE)')
    legend('Filter baseline','Smoother baseline','Filter learn', 'Smoother learn','Ensemble Filter Learn', 'Ensemble Smoother Learn', 'Location', 'best')
    title('User Factor Prediction (U)')
    
    % plot preference prediction
    figure('Position', [100 100 600 500], 'Color', 'w');
    hold all
    plot(1:iter, repmat(rmsesvd,iter,1), '-', 'LineWidth',lineWidth, 'Color', ColorSet(7,:))
    plot(1:iter, rmseY_kal_base*ones(iter,1), '-','LineWidth',lineWidth, 'Color', ColorSet(1,:))
    plot(1:iter, rmseY_rts_base*ones(iter,1), '-','LineWidth',lineWidth, 'Color', ColorSet(2,:))
    plot(1:iter, rmseY_kal_learn, '-','LineWidth',lineWidth, 'Color', ColorSet(3,:))
    plot(1:iter, rmseY_rts_learn, '-','LineWidth',lineWidth, 'Color', ColorSet(4,:))    
    plot(1:iter, rmseY_kal_learnEn, '-','LineWidth',lineWidth, 'Color', ColorSet(5,:))
    plot(1:iter, rmseY_rts_learnEn, '-', 'LineWidth',lineWidth,'Color', ColorSet(6,:))   
    
    xlabel('Iteration')
    ylabel('Root Mean Square Error (RMSE)')
    legend('Objective Function','Filter baseline','Smoother baseline','Filter learn', 'Smoother learn','Ensemble Filter Learn', 'Ensemble Smoother Learn', 'Location', 'best')
    title('Observation Preference Estimation (Y)')      
    hold off
    shg
    end
end

% Get RMSE for matrix
function out = myRMSE( in )
    out = sqrt(sum(in(:).^2)/length(in(:)));
end

%------------- END OF CODE --------------
