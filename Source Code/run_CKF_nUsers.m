% RUN_CKF_NUSERS - run Kalman filter/smoother on all N users
%
% Syntax: [x_at0 x_at0 x_pos x_tot P_out P_pri P_pos P_tot P_lag] = run_CKF_nUsers(param, Y)
%
% Inputs:
%    param  - parameters for Kalman filter/smoother
%    Y      - observed measurements
%
% Outputs:
%    [x_pri P_pri]  - a priori estimates and covariances
%    [x_pos P_pos]  - a posteriori estimates and covariances
%    [x_tot P_tot]  - smoothed estimates and and covariances
%    [x_at0 P_at0]  - prediction of x_0 and covariance
%    [P_lag]        - lag covariance
%
%
% Author: John Z. Sun
% IBM/MIT
% email: johnsun@mit.edu
% Aug 2011; Last revision: 10-31-2012

%------------- BEGIN CODE --------------

function [x_at0 x_pri x_pos x_tot P_at0 P_pri P_pos P_tot P_lag] ...
                                            = run_CKF_nUsers(param, Y)

    % initialize
    N = param.N;
    x_at0 = cell(N,1);
    x_pri = cell(N,1);
    x_pos = cell(N,1);
    x_tot = cell(N,1);
    P_at0 = cell(N,1);
    P_pri = cell(N,1);
    P_pos = cell(N,1);
    P_tot = cell(N,1);
    P_lag = cell(N,1);
    
    % run Kalman filter/smoother on each user
    for n = 1:N
        [mu_pri Cov_pri mu_pos Cov_pos mu_tot Cov_tot Cov_lag mu_hat Cov_hat] = run_CKF_nFBKF(param, Y, n);
        x_at0{n} = mu_hat;
        x_pri{n} = mu_pri;
        x_pos{n} = mu_pos;
        x_tot{n} = mu_tot;
        P_at0{n} = Cov_hat;
        P_pri{n} = Cov_pri;
        P_pos{n} = Cov_pos;
        P_tot{n} = Cov_tot;
        P_lag{n} = Cov_lag;
    end
end

%------------- END OF CODE --------------