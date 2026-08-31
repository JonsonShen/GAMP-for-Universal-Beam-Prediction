function [x_opt] = active_set_method(H_qua, f_qua, Aeq_qua, beq_qua)
    [~, ~, E] = qr(Aeq_qua', 'vector');
    rank_Aeq_qua = rank(Aeq_qua);
    ind_rows = E(1:rank_Aeq_qua);
    Aeq_qua_new = Aeq_qua(ind_rows, :);
    beq_qua_new = beq_qua(ind_rows);
    
    tol = 1e-4;
    options = optimset('TolX', 1e-4, 'MaxIter', 500);
    x0 = lsqnonneg(Aeq_qua_new, beq_qua_new, options);
    if norm(Aeq_qua_new*x0-beq_qua_new) >= tol
        x_opt = 1;
        return;
    end

    KKT = [H_qua, Aeq_qua_new'; Aeq_qua_new, zeros(size(Aeq_qua_new, 1), size(Aeq_qua_new, 1))];
    sol = KKT \ [-f_qua; beq_qua_new];
    r_k = sol(1:length(f_qua), 1);

    if (all(r_k > 0))
        x_opt = r_k;
        return
    end

    A_all = [Aeq_qua_new; -1*eye(length(f_qua))];
    b_all = [beq_qua_new; -0.0*ones(length(f_qua), 1)];
    xk = x0;
    vals = A_all * xk;
    Wk = find(abs(vals - b_all) < tol);
    Wk = sort(Wk);

    n = length(xk);
    m = length(beq_qua_new);
    k = 0;

    while k<=100
        k = k+1;
        gk = H_qua*xk + f_qua;
        A = A_all(Wk, :);
        KKT = [H_qua, A'; A, zeros(length(Wk), length(Wk))];
        rhs = [-gk; zeros(length(Wk),1)];

        sol = KKT \ rhs;
        dk = sol(1:n);
        lambda_Wk = sol(n+1:end);

        if norm(dk) < tol
            if isempty(lambda_Wk(m+1:end)) || all(lambda_Wk(m+1:end) >= -tol)
                x_opt = xk;
                return;
            else
                [~, idx_min] = min(lambda_Wk(m+1:end));
                Wk(idx_min+m) = [];
            end
        else
            alpha = 1;
            blocking_idx = -1;
            for i = 1:size(A_all,1)
                if ~ismember(i, Wk)
                    ai = A_all(i,:)';
                    if ai' * dk > tol
                        alpha_i = (b_all(i) - ai' * xk) / (ai' * dk);
                        if alpha_i < alpha
                            alpha = alpha_i;
                            blocking_idx = i;
                        end
                    end
                end
            end

            xk = xk + alpha * dk;

            if alpha < 1 - tol && blocking_idx > 0
                Wk = [Wk; blocking_idx];
                Wk = sort(Wk);
            end
        end
    end
    x_opt = xk;
    return;
end