function [rK2_opt, rK0Rx_pred] = case2_optimization(rK2_0, muK2_1, SigmaK2_1, muK2_0, muK0_Tx, SigmaK0_K2, SigmaK2_Tx, muK2_Tx)
    % Input:
    % rK2_0: Observed rK2
    % muK2_1: Mean of rK2,1
    % SigmaK2_1: Covariance matrix of rK2,1
    % muK2_0: Mean of rK2,0
    % muK0_Tx: Mean of rK0,Tx
    % SigmaK0_K2: Covariance matrix between K0 and K2
    % SigmaK2_Tx: Covariance matrix of K2,Tx
    % muK2_Tx: Mean of K2,Tx

    % Objective function: Minimize the likelihood error
    objective = @(rK2_1) (rK2_1 - muK2_1)' * inv(SigmaK2_1) * (rK2_1 - muK2_1);

    % Initial guess for rK2_1
    rK2_1_init = muK2_1;

    % Optimization options : using Sequential quadratic programming
    % sqp or interior-point?

    options = optimoptions('fmincon', 'Algorithm', 'sqp');

    % Upper bound and lower bound? x


    % Constraint function 
    constraint = @(rK2_1) deal([], muK0_Tx + SigmaK0_K2 * inv(SigmaK2_Tx) * ([rK2_0; rK2_1] - muK2_Tx));

    % Solve the optimization problem
    % fmincon : Find minimum of constrained nonlinear multivariable function
    % fmincon(objective function, initial guess, linear inequality constraints, linear equality constraints,
    % lower bounds for nonlinear inequality, upper bounds for nonlinear inequality,
    % lower bounds for the variables, upper bounds for the variable, 
    % nonlinear constraints(inequality and equality), options)
    [rK2_opt, ~] = fmincon(objective, rK2_1_init, [], [], [], [], [], [], constraint, options);

    
    % Compute predicted rK0,Rx based on optimized rK2_1
    rK0Rx_pred = muK0_Tx + SigmaK0_K2 * inv(SigmaK2_Tx) * ([rK2_0; rK2_opt] - muK2_Tx);
end
