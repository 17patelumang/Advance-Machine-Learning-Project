AML PROJECT - UMANG PATEL and ILAMBHARATHI KANNIAH

Contact ujp2001@columbia.edu or ik2342@columbia.edu for questions.


Command to run the DEMO:
demo


Dependencies required for this project to RUN:
------------------------
CVX : Package for Disciplined Convex Programming
http://cvxr.com/cvx/

The files included are :
------------------------
demo.m 									- Runs the wrapper.m and clustering(5) for demo purposes.


wrapper.m 								- Runs the extended model having KF-KS, EnKF-EnKS and Initial Clustering of Users.

run_CKF_EM.m, run_CKF_EM_EnKF			- runs EM learning algorithm

run_CKF_nUsers.m, run_CKF_nUsers_EM_O.m	- wrapper for the n-user Kalman filter and Ensemble Kalman Filter respectively.

run_CKF_nFBKF.m, run_CKF_nFBKF_enKF		- forward/backward algorithm to implement Kalman filter/smoother and Ensemble Kalman filter/ Ensemble Kalman smoother.

plot_Results.m							- plot results

generate_CKF_data.m						- generate the gold data - output to data_CKF_test

stograd_func.m   						- Stochastic gradient method to directly solve the minimization problem.

svdtimin.m 								- Helper function to run alternating lock minimization solution using CVX solver.

clustering.m 							- Runs the initial clustering of users into user-clusters and runs the model completely for each user-cluster.

Main instructions:
------------------
demo.m

	Part-1
	
    Runs wrapper.m to implement learning and a baseline comparison.
    
	1) Loads data, user specifies what parameters are unknown
	
	2) Runs a baseline test of estimating user factor tensor assuming all model parameters and item factor matrix is known.  This is a lower bound for the achievable prediction performance.
	
	3) Runs EM algorithm to learn both model parameters, item factor matrix and user factor tensor.
	
	4) Plot comparison between actual prediction performance and baseline.
	
	Part-2
	
	Runs clustering of users into user-clusters and runs the wrapper.m for each of the clusters.
	
	
If you want to generate a different dataset,

    Run generate_CKF_data.m to create dataset based on state space model for parameters that the user specifies.
    
	This generates both a true preference tensor as well as a training set of observed preferences. 
	
	The resulting parameters and preferences are saved to data_CKF_test
	
