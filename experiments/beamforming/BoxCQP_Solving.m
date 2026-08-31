function [r_N, lambda_U] = BoxCQP_Solving(r_N, lambda_U, Q_NN, Q_UN, c_N, c_U, A_N, A_U, b)
    
    matrix = [Q_UN, A_U', -eye(length(lambda_U), length(lambda_U)); Q_NN, A_N', zeros(length(r_N), length(lambda_U)); A_N, zeros(length(b), length(b)), zeros(length(b), length(lambda_U))];
    sol = matrix \ [-c_U; -c_N; zeros(length(b), 1)];
    r_N = sol(1:length(r_N), 1);
    lambda_U = sol(length(r_N)+length(b)+1:end, 1);
    return
end