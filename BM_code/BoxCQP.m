function [r_opt, fea ,test, one] = BoxCQP(Q, c, A, b)

    fea = 0;
    test = 0;
    one = 0;
    tol = 1e-4;
    k = 1;

    [~, ~, E] = qr(A', 'vector');
    rank_A = rank(A);
    ind_rows = E(1:rank_A);
    A_fr = A(ind_rows, :);
    b_fr = b(ind_rows, 1);

    options = optimset('TolX', 1e-4, 'MaxIter', 500);
    r_0 = lsqnonneg(A_fr, b_fr, options);
    if (norm(A_fr*r_0 - b_fr) > tol)
        fea = 1;
        r_opt = r_0;
        return
    end
    
    KKT = [Q, A_fr'; A_fr, zeros(size(A_fr, 1), size(A_fr, 1))];
    sol = KKT \ [-c; b_fr];
    r_k = sol(1:length(c), 1);
    r_opt = r_k;
    lambda_k = zeros(size(r_k));

    if (all(r_k > 0))
        one = 1;
        return
    end

    while k <= 100
        L_k = zeros(length(r_k), 1);
        S_k = zeros(length(r_k), 1);
        for i = 1:length(r_k)
            if (r_k(i) < 0) || (lambda_k(i) > 0)
                L_k(i) = 1;
            else
                S_k(i) = 1;
            end
        end

        idx = [find(S_k == 1); find(S_k == 0)];

        Q = Q(idx, :);
        c = c(idx, 1);
        A = A(:, idx);

        r_N = ones(sum(S_k == 1), 1);

        if (length(r_N) >= length(b))
            lambda_U = ones(sum(L_k == 1), 1);
            Q_NN = Q(1:length(r_N), 1:length(r_N));
            Q_UN = Q(length(r_N)+1:end, 1:length(r_N));
            c_N = c(1:length(r_N), 1);
            c_U = c(length(r_N)+1:end, 1);
            A_N = A(:, 1:length(r_N));
            A_U = A(:, length(r_N)+1:end);

            [r_N, lambda_U] = BoxCQP_Solving(r_N, lambda_U, Q_NN, Q_UN, c_N, c_U, A_N, A_U, b);
        else
            % A_Ac = A(1:length(r_N), 1:length(r_N));

            test = 1;
            r_opt = r_k;
            return
        end
        
        r_k = [r_N; zeros(sum(L_k == 1), 1)];
        lambda_k = [zeros(sum(S_k == 1), 1); lambda_U];

        r_k_temp = zeros(size(r_k));
        lambda_k_temp = zeros(size(lambda_k));
        Q_temp = zeros(size(Q));
        c_temp = zeros(size(c));
        A_temp = zeros(size(A));

        r_k_temp(idx, 1) = r_k;
        lambda_k_temp(idx, 1) = lambda_k;
        Q_temp(idx, :) = Q;
        c_temp(idx, 1) = c;
        A_temp(:, idx) = A;

        r_k = r_k_temp;
        lambda_k = lambda_k_temp;
        Q = Q_temp;
        c = c_temp;
        A = A_temp;

        if (all(r_N > 0)) && (all(lambda_U >= 0))
            r_opt = r_k;
            return
        end

        k = k+1;
    end


    
end