tic
mean_yn = [10, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1];
lambda_yn = ones(size(mean_yn)) ./ mean_yn;
kn = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
gain_1 = [kn(1), kn(2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
gain_2 = [0, 0, kn(3), kn(4), 0, 0, 0, 0, 0, 0, 0, 0];
gain_3 = [0, 0, 0, 0, kn(5), kn(6), 0, 0, 0, 0, 0, 0];
gain_4 = [0, 0, 0, 0, 0, 0, kn(7), kn(8), 0, 0, 0, 0];
gain_5 = [0, 0, 0, 0, 0, 0, 0, 0, kn(9), kn(10), 0, 0];
gain_6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kn(11), kn(12)];
gain = [gain_1; gain_2; gain_3; gain_4; gain_5; gain_6];

sample = zeros(1000000, 18);

for i = 1:1000000
    sample(i, 1:12) = exprnd(mean_yn);
    for j = 1:6
        sample(i, j+12) = sum(sample(i, 1:12).*gain(j, 1:12));
    end
end

sample_mean = mean(sample);
sample_cov = cov(sample);

test_num = 100000;
y_all = zeros(test_num, 18);

real_max = zeros(18, 1);

prd_lmmse_max = zeros(18, 1);
error_lmmse = 0;

prd_opt_max = zeros(18, 1);
error_opt = 0;

snr_dB = 20;

for i = 1:test_num
    yn = exprnd(mean_yn);
    beam = gain*yn';
    for j = 1:6
        beam(j) = beam(j) + normrnd(0, sample_cov(j+12, j+12)/(10^(snr_dB/10)));
    end
    y_all(i, :) = [yn, beam'];

    prd_lmmse = zeros(1, 18);
    prd_opt = zeros(1, 18);

    prd_lmmse(1, 13:18) = beam';
    for j = 1:6
        prd_lmmse(2*j-1) = sample_mean(2*j-1)+sample_cov(2*j-1, j+12)/ ...
            sample_cov(j+12, j+12)*(beam(j)-sample_mean(j+12));
        prd_lmmse(2*j) = sample_mean(2*j)+sample_cov(2*j, j+12)/ ...
            sample_cov(j+12, j+12)*(beam(j)-sample_mean(j+12));
    end
    
    for k = 1:12
        num = round(k/2);
        
        lambda_temp = ones(1, 12) ./ sample_mean(1, 1:12);
        lambda_temp(2*num) = [];
        lambda_temp(2*num-1) = [];
        if (mod(k, 2) == 0)
            lambda = [lambda_yn(k), lambda_yn(k-1), lambda_temp];
        else
            lambda = [lambda_yn(k), lambda_yn(k+1), lambda_temp];
        end

        ki_temp = kn;
        ki_temp(2*num) = [];
        ki_temp(2*num-1) = [];
        if (mod(k, 2) == 0)
            ki = [kn(k), kn(k-1), ki_temp];
        else
            ki = [kn(k), kn(k+1), ki_temp];
        end

        H_temp = beam;
        H_temp(num) = [];
        H = [beam(num); H_temp];

        alpha = zeros(6, 1);

        for l = 1:6
            alpha(l) = -(lambda(2*l-1)-ki(2*l-1)/ki(2*l)*lambda(2*l));
        end

        integrand = @(y1) ...
            exp(alpha(1).*y1) .* exp(-lambda(2)*H(1)/ki(2)) ...
            ./ alpha(2) .* exp(-lambda(4)*H(2)/ki(4)) ...
            .* (exp(alpha(2).*min(H(2)/ki(3), y1)) - exp(alpha(2).*max(0, (H(2)-ki(4).*y1)/ki(3)))) ...
            ./ alpha(3) .* exp(-lambda(6)*H(3)/ki(6)) ...
            .* (exp(alpha(3).*min(H(3)/ki(5), y1)) - exp(alpha(3).*max(0, (H(3)-ki(6).*y1)/ki(5)))) ...
            ./ alpha(4) .* exp(-lambda(8)*H(4)/ki(8)) ...
            .* (exp(alpha(4).*min(H(4)/ki(7), y1)) - exp(alpha(4).*max(0, (H(4)-ki(8).*y1)/ki(7)))) ...
            ./ alpha(5) .* exp(-lambda(10)*H(5)/ki(10)) ...
            .* (exp(alpha(5).*min(H(5)/ki(9), y1)) - exp(alpha(5).*max(0, (H(5)-ki(10).*y1)/ki(9)))) ...
            ./ alpha(6) .* exp(-lambda(12)*H(6)/ki(12)) ...
            .* (exp(alpha(6).*min(H(6)/ki(11), y1)) - exp(alpha(6).*max(0, (H(6)-ki(12).*y1)/ki(11))));
        
        y1_lower = max([H(1)/(ki(1)+ki(2)), H(2)/(ki(3)+ki(4)), H(3)/(ki(5)+ki(6)), H(4)/(ki(7)+ki(8)), H(5)/(ki(9)+ki(10)), H(6)/(ki(11)+ki(12)), H(1), H(2), H(3), H(4), H(5), H(6)]);
        y1_upper = H(1)/ki(1);

        if (y1_upper > y1_lower)
            result = integral(@(y1) integrand(y1), y1_lower, y1_upper);
        else
            result = 0;
        end

        prd_opt(k) = result;
    end

    for k = 1:6
        num = k;
        
        lambda_temp = ones(1, 12) ./ sample_mean(1, 1:12);
        lambda_temp(2*num) = [];
        lambda_temp(2*num-1) = [];
        lambda = [lambda_yn(2*k-1), lambda_yn(2*k), lambda_temp];

        ki_temp = kn;
        ki_temp(2*num) = [];
        ki_temp(2*num-1) = [];
        ki = [kn(2*k-1), kn(2*k), ki_temp];

        H_temp = beam;
        H_temp(num) = [];
        H = [beam(num); H_temp];

        alpha = zeros(6, 1);

        for l = 1:6
            alpha(l) = -(lambda(2*l-1)-ki(2*l-1)/ki(2*l)*lambda(2*l));
        end

        pre_integrand = 1 ./ alpha(2) .* exp(-lambda(4)*H(2)/ki(4)) ...
            .* (exp(alpha(2).*min(H(2)/ki(3), H(1))) - exp(alpha(2).*max(0, (H(2)-ki(4).*H(1))/ki(3)))) ...
            ./ alpha(3) .* exp(-lambda(6)*H(3)/ki(6)) ...
            .* (exp(alpha(3).*min(H(3)/ki(5), H(1))) - exp(alpha(3).*max(0, (H(3)-ki(6).*H(1))/ki(5)))) ...
            ./ alpha(4) .* exp(-lambda(8)*H(4)/ki(8)) ...
            .* (exp(alpha(4).*min(H(4)/ki(7), H(1))) - exp(alpha(4).*max(0, (H(4)-ki(8).*H(1))/ki(7)))) ...
            ./ alpha(5) .* exp(-lambda(10)*H(5)/ki(10)) ...
            .* (exp(alpha(5).*min(H(5)/ki(9), H(1))) - exp(alpha(5).*max(0, (H(5)-ki(10).*H(1))/ki(9)))) ...
            ./ alpha(6) .* exp(-lambda(12)*H(6)/ki(12)) ...
            .* (exp(alpha(6).*min(H(6)/ki(11), H(1))) - exp(alpha(6).*max(0, (H(6)-ki(12).*H(1))/ki(11))));
        integrand = @(y1) exp(alpha(1).*y1) .* exp(-lambda(2)*H(1)/ki(2)) .* pre_integrand;
        
        y1_lower = max([0, (1-ki(2))/ki(1)*H(1)]);
        y1_upper = min([H(1), H(1)/ki(1)]);

        if (y1_upper > y1_lower) && ((ki(1)+ki(2)) >= 1)...
                && (H(1) >= max([H(2)/(ki(3)+ki(4)), H(3)/(ki(5)+ki(6)), H(4)/(ki(7)+ki(8)), H(5)/(ki(9)+ki(10)), H(6)/(ki(11)+ki(12)), H(2), H(3), H(4), H(5), H(6)]))
            result = integral(@(y1) integrand(y1), y1_lower, y1_upper);
        else
            result = 0;
        end

        prd_opt(k+12) = result;
    end

    real_y = [yn, beam'];
    [~, x] = max(real_y);
    [~, y] = max(prd_lmmse);
    [~, z] = max(prd_opt);
    real_max(x) = real_max(x)+1;
    prd_lmmse_max(y) = prd_lmmse_max(y)+1;
    prd_opt_max(z) = prd_opt_max(z)+1;
    if (x ~= y)
        error_lmmse = error_lmmse+1;
    end
    if (x ~= z)
        error_opt = error_opt+1;
    end
end
toc