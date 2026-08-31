%%

tic
mean_h = [10, 1, 2, 1, 2, 1, 2, 1];
lambda_h = ones(1, 8) ./ mean_h;
k_i = [0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2];
gain_1 = [k_i(1), k_i(2), 0, 0, 0, 0, 0, 0];
gain_2 = [0, 0, k_i(3), k_i(4), 0, 0, 0, 0];
gain_3 = [0, 0, 0, 0, k_i(5), k_i(6), 0, 0];
gain_4 = [0, 0, 0, 0, 0, 0, k_i(7), k_i(8)];
    
sample = zeros(100000, 12);
sample_max = zeros(8, 1);

for a = 1:100000
    sample(a, 1:8) = exprnd(mean_h);
    sample(a, 9) = sum(sample(a, 1:8).*gain_1);
    sample(a, 10) = sum(sample(a, 1:8).*gain_2);
    sample(a, 11) = sum(sample(a, 1:8).*gain_3);
    sample(a, 12) = sum(sample(a, 1:8).*gain_4);
    [~, x] = max(sample(a, 1:8));
    sample_max(x) = sample_max(x) + 1;
end

sample_mean = mean(sample);
sample_cov = cov(sample);

h_num = 100000;
prd_max = zeros(8, 1);
prd_beam = zeros(4, 1);
real_max = zeros(8, 1);
real_beam = zeros(4, 1);
error = 0;
error_beam = 0;
h_all = zeros(h_num, 8);

prd_max_ = zeros(8, 1);
prd_beam_ = zeros(4, 1);
error_ = 0;
error_beam_ = 0;

prd_max_e = zeros(8, 1);
prd_beam_e = zeros(4, 1);
error_e = 0;
error_beam_e = 0;

prd_max_o = zeros(8, 1);
prd_beam_o = zeros(4, 1);
error_o = 0;
error_beam_o = 0;

error_num_o = 0; % gaussian correct, optimal wrong
error_num = 0; % gaissian wrong, optimal correct

for i = 1:h_num
    h = exprnd(mean_h);
    h_all(i, :) = h;
    beam = zeros(1, 4);

    temp = h.*gain_1;

    beam(1) = sum(h.*gain_1);
    beam(2) = sum(h.*gain_2);
    beam(3) = sum(h.*gain_3);
    beam(4) = sum(h.*gain_4);

    prd = zeros(8, 1);

    prd(1) = sample_mean(1)+sample_cov(1, 9)/sample_cov(9, 9)*(beam(1)-sample_mean(9));
    prd(2) = sample_mean(2)+sample_cov(2, 9)/sample_cov(9, 9)*(beam(1)-sample_mean(9));
    prd(3) = sample_mean(3)+sample_cov(3, 10)/sample_cov(10, 10)*(beam(2)-sample_mean(10));
    prd(4) = sample_mean(4)+sample_cov(4, 10)/sample_cov(10, 10)*(beam(2)-sample_mean(10));
    prd(5) = sample_mean(5)+sample_cov(5, 11)/sample_cov(11,11)*(beam(3)-sample_mean(11));
    prd(6) = sample_mean(6)+sample_cov(6, 11)/sample_cov(11, 11)*(beam(3)-sample_mean(11));
    prd(7) = sample_mean(7)+sample_cov(7, 12)/sample_cov(12, 12)*(beam(4)-sample_mean(12));
    prd(8) = sample_mean(8)+sample_cov(8, 12)/sample_cov(12, 12)*(beam(4)-sample_mean(12));

    %prd = prd ./ transpose(k_i);

    prd_ = zeros(8, 1);

    prd_(1) = beam(1);
    prd_(3) = beam(2);
    prd_(5) = beam(3);
    prd_(7) = beam(4);

    %prd_ = prd_ ./ transpose(k_i);

    prd_e = zeros(8, 1);

    e1 = zeros(4, 1);
    e2 = zeros(4, 1);
    for j = 1:4
        e1(j) = exp(beam(j)*lambda_h(2*j-1));
        e2(j) = exp(beam(j)*lambda_h(2*j));

        prd_e(2*j-1) = -beam(j)*lambda_h(2*j-1)*e2(j)+beam(j)*lambda_h(2*j)*e2(j)+e1(j)-e2(j);
        prd_e(2*j) = -beam(j)*lambda_h(2*j)*e1(j)+beam(j)*lambda_h(2*j-1)*e1(j)+e2(j)-e1(j);
        prd_e(2*j-1) = prd_e(2*j-1)/(lambda_h(2*j-1)*e1(j)+lambda_h(2*j)*e2(j)-lambda_h(2*j-1)*e2(j)-lambda_h(2*j)*e1(j));
        prd_e(2*j) = prd_e(2*j)/(lambda_h(2*j-1)*e1(j)+lambda_h(2*j)*e2(j)-lambda_h(2*j-1)*e2(j)-lambda_h(2*j)*e1(j));
    end

    %prd_e = prd_e ./ transpose(k_i);

    prd_o = zeros(8, 1);

    for k = 1:8
        num = round(k/2);
        % e_k = -lambda_h(k)*beam(num);
        % prd_o(k) = prd_o(k) + 1/(lambda_h(k))*(exp(e_k/2)-exp(e_k));
        % for l = 1:8
        %     if (l ~= k)
        %         sum_l = lambda_h(k)+lambda_h(l);
        %         e_l = -sum_l*beam(num);
        %         prd_o(k) = prd_o(k) - 1/(sum_l)*(exp(e_l/2)-exp(e_l));
        %     end
        %     for m = 1:8
        %         if (l ~= k) && (m ~= k) && (l ~= m)
        %             sum_m = lambda_h(k)+lambda_h(l)+lambda_h(m);
        %             e_m = -sum_m*beam(num);
        %             prd_o(k) = prd_o(k) + 1/(sum_m)*(exp(e_m/2)-exp(e_m));
        %         end
        %         for n = 1:8
        %             if (n ~= k) && (m ~= k) && (l ~= k) && (n ~= m) && (n ~= l) && (m ~= l)
        %                 sum_n = lambda_h(k)+lambda_h(l)+lambda_h(m)+lambda_h(n);
        %                 e_n = -sum_n*beam(num);
        %                 prd_o(k) = prd_o(k) - 1/(sum_n)*(exp(e_n/2)-exp(e_n));
        %             end
        %         end
        %     end
        % end
        % prd_o(k) = prd_o(k)*lambda_h(k);

        % H = beam(num);
        % lambda_y1 = lambda_h(k);
        % lambda_vec = lambda_h;
        % lambda_vec(lambda_vec == lambda_y1) = [];
        % 
        % f_integrand = @(y1) lambda_y1 * exp(-lambda_y1 * y1) .* ...
        % prod(1 - exp(-lambda_vec(:) * y1), 1);
        % 
        % prd_o(k) = integral(f_integrand, H/2, H);
        % 
        % f = lambda_h(num*2)*lambda_h(num*2-1)/(lambda_h(num*2)-lambda_h(num*2-1))*(exp(-lambda_h(num*2-1)*beam(num))-exp(-lambda_h(num*2)*beam(num)));
        % prd_o(k) = prd_o(k)/f;
        
        lambda_temp = lambda_h;
        lambda_temp(2*num) = [];
        lambda_temp(2*num-1) = [];
        if (mod(k, 2) == 0)
            lambda = [lambda_h(k), lambda_h(k-1), lambda_temp];
        else
            lambda = [lambda_h(k), lambda_h(k+1), lambda_temp];
        end

        ki_temp = k_i;
        ki_temp(2*num) = [];
        ki_temp(2*num-1) = [];
        if (mod(k, 2) == 0)
            ki = [k_i(k), k_i(k-1), ki_temp];
        else
            ki = [k_i(k), k_i(k+1), ki_temp];
        end

        H_temp = beam;
        H_temp(num) = [];
        H = [beam(num), H_temp];

        % B12 = beta(lambda(1), lambda(2));
        % B34 = beta(lambda(3), lambda(4));
        % B56 = beta(lambda(5), lambda(6));
        % B78 = beta(lambda(7), lambda(8));
        % 
        % g = @(x1) ...
        %     (x1.^(lambda(1)-1) .* (1 - x1).^(lambda(2)-1)) / beta(lambda(1), lambda(2)) .* ...
        %     (betainc(min(1,x1*H(1)/H(2)), lambda(3), lambda(4)) - betainc(max(0,1 - x1*H(1)/H(2)), lambda(3), lambda(4))) .* ...
        %     (betainc(min(1,x1*H(1)/H(3)), lambda(5), lambda(6)) - betainc(max(0,1 - x1*H(1)/H(3)), lambda(5), lambda(6))) .* ...
        %     (betainc(min(1,x1*H(1)/H(4)), lambda(7), lambda(8)) - betainc(max(0,1 - x1*H(1)/H(4)), lambda(7), lambda(8)));
        % 
        % if (mod(k, 2) == 0)
        %     prd_o(k-1) = integral(g, 0.5, 1-1e-6);
        % else
        %     prd_o(k+1) = integral(g, 0.5, 1-1e-6);
        % end

        alpha1 = -2*(lambda(1)-ki(1)/ki(2)*lambda(2));
        alpha2 = -2*(lambda(3)-ki(3)/ki(4)*lambda(4));
        alpha3 = -2*(lambda(5)-ki(5)/ki(6)*lambda(6));
        alpha4 = -2*(lambda(7)-ki(7)/ki(8)*lambda(8));

        integrand = @(y1) ...
            exp(alpha1.*y1) ...
            ./ alpha2 ...
            .* (exp(alpha2.*min(H(2)/ki(3), y1)) - exp(alpha2.*max(0, (H(2)-ki(4).*y1)/ki(3)))) ...
            ./ alpha3 ...
            .* (exp(alpha3.*min(H(3)/ki(5), y1)) - exp(alpha3.*max(0, (H(3)-ki(6).*y1)/ki(5)))) ...
            ./ alpha4 ...
            .* (exp(alpha4.*min(H(4)/ki(7), y1)) - exp(alpha4.*max(0, (H(4)-ki(8).*y1)/ki(7))));
        
        y1_lower = max([H(1)/(ki(1)+ki(2)), H(2)/(ki(3)+ki(4)), H(3)/(ki(5)+ki(6)), H(4)/(ki(7)+ki(8))]);
        y1_upper = H(1)/ki(1);

        if (y1_upper > y1_lower)
            result = integral(@(y1) integrand(y1), y1_lower, y1_upper);
        else
            result = 0;
        end

        % if (mod(k, 2) == 0)
        %     prd_o(k-1) = result;
        % else
        %     prd_o(k+1) = result;
        % end

        prd_o(k) = result;
    end

    h_new = h;

    [~, x] = max(h_new);
    X = round(x/2);
    [~, y] = max(prd);
    Y = round(y/2);
    [~, z] = max(prd_);
    Z = round(z/2);
    [~, w] = max(prd_e);
    W = round(w/2);
    [~, q] = max(prd_o);
    Q = round(q/2);
    real_max(x) = real_max(x) + 1;
    real_beam(X) = real_beam(X) + 1;
    prd_max(y) = prd_max(y) + 1;
    prd_beam(Y) = prd_beam(Y) + 1;
    prd_max_(z) = prd_max_(z) + 1;
    prd_beam_(Z) = prd_beam_(Z) + 1;
    prd_max_e(w) = prd_max_e(w) + 1;
    prd_beam_e(W) = prd_beam_e(W) + 1;
    prd_max_o(q) = prd_max_o(q) + 1;
    prd_beam_o(Q) = prd_beam_o(Q) + 1;
    if (nnz(x-y) ~= 0)
        error = error+1;
    end
    if (nnz(X-Y) ~= 0)
        error_beam = error_beam+1;
    end
    if (nnz(x-z) ~= 0)
        error_ = error_+1;
    end
    if (nnz(X-Z) ~= 0)
        error_beam_ = error_beam_+1;
    end
    if (nnz(x-w) ~= 0)
        error_e = error_e+1;
    end
    if (nnz(X-W) ~= 0)
        error_beam_e = error_beam_e+1;
    end
    if (nnz(x-q) ~= 0)
        error_o = error_o+1;
    end
    if (nnz(X-Q) ~= 0)
        error_beam_o = error_beam_o+1;
    end

    if (nnz(x-q) > nnz(x-y))
        error_num_o = error_num_o + 1;
    elseif (nnz(x-y) > nnz(x-q))
        error_num = error_num + 1;
    end
end

error_total = [error, error_, error_e, error_o; error_beam, error_beam_, error_beam_e, error_beam_o];
error_total = error_total ./ h_num .* 100;

toc
%%

% mean_1 = 10;
% mean_2 = 1;
% 
% lambda_1 = 1/mean_1;
% lambda_2 = 1/mean_2;
% 
% N = 1e6;
% 
% X1 = exprnd(1/lambda_1, N, 1);
% X2 = exprnd(1/lambda_2, N, 1);
% 
% mask1 = X2 < X1;
% X1_cond1 = X1(mask1);
% X2_cond1 = X2(mask1);
% mask2 = X2 > X1;
% X1_cond2 = X1(mask2);
% X2_cond2 = X2(mask2);
% 
% ratio = length(X1_cond1)/N; % prob of X2>X1
% 
% threshold = 2/(lambda_1+lambda_2);
% 
% prob = (sum((X1_cond1+X2_cond1)<threshold)+sum((X1_cond2+X2_cond2)>threshold))/N;

%%

% x = linspace(0 ,10);
% y = (lambda_1*lambda_2/(lambda_2-lambda_1))*(exp(-lambda_1*x)-exp(-lambda_2*x));
% k = 1/(lambda_1+lambda_2);
% figure(1)
% plot(x, y)
% hold on
% plot(ones(size(x)).*k, linspace(0, 0.1))