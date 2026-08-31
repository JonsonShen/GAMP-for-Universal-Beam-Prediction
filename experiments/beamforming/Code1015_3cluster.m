mean_yn = [15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
lambda_yn = ones(size(mean_yn)) ./ mean_yn;
kn = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
gain_1 = [kn(1), kn(2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
gain_2 = [0, 0, kn(3), kn(4), 0, 0, 0, 0, 0, 0, 0, 0];
gain_3 = [0, 0, 0, 0, kn(5), kn(6), 0, 0, 0, 0, 0, 0];
gain_4 = [0, 0, 0, 0, 0, 0, kn(7), kn(8), kn(9), 0, 0, 0];
gain_5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, kn(10), kn(11), kn(12)];
gain = [gain_1; gain_2; gain_3; gain_4; gain_5];

sample = zeros(10000, 12+5);

for i = 1:10000
    sample(i, 1:12) = exprnd(mean_yn);
    for j = 1:5
        sample(i, j+12) = sum(sample(i, 1:12).*gain(j, 1:12));
    end
end

sample_mean = mean(sample);
sample_cov = cov(sample);

test_num = 100000;
y_all = zeros(test_num, 12+5);

real_max = zeros(12, 1);

prd_lmmse_max = zeros(12+5, 1);
error_lmmse = 0;

prd_opt_max = zeros(12+5, 1);
error_opt = 0;

for i = 1:test_num
    yn = exprnd(mean_yn);
    beam = gain*yn';
    y_all(i, :) = [yn, beam'];

    prd_lmmse = zeros(1, 17);
    prd_opt = zeros(1, 17);
    prd_lmmse(1, 13:17) = beam';
    prd_opt(1, 13:17) = beam';

    for j = 1:3
        prd_lmmse(2*j-1) = sample_mean(2*j-1)+sample_cov(2*j-1, j+12)/ ...
            sample_cov(j+12, j+12)*(beam(j)-sample_mean(j+12));
        prd_lmmse(2*j) = sample_mean(3+2*j)+sample_cov(3+2*j, j+12)/ ...
            sample_cov(j+12, j+12)*(beam(j)-sample_mean(j+12));
    end
    for j = 1:2
        prd_lmmse(3*j+4) = sample_mean(3*j+4)+sample_cov(3*j+4, j+15)/ ...
            sample_cov(j+15, j+15)*(beam(j+3)-sample_mean(j+15));
        prd_lmmse(3*j+5) = sample_mean(3*j+5)+sample_cov(3*j+5, j+15)/ ...
            sample_cov(j+15, j+15)*(beam(j+3)-sample_mean(j+15));
        prd_lmmse(3*j+6) = sample_mean(3*j+6)+sample_cov(3*j+6, j+15)/ ...
            sample_cov(j+15, j+15)*(beam(j+3)-sample_mean(j+15));
    end

    for j = 1:6
        
    end

    real_y = [yn./kn, beam'];
    [~, x] = max(real_y);
    [~, y] = max(prd_lmmse);
    [~, z] = max(prd_opt);
    real_max(x) = real_max(x)+1;
    prd_lmmse_max(y) = prd_lmmse_max(y)+1;
    prd_opt_max(z) = prd_opt_max(z)+1;
    if (nnz(x-y) ~= 0)
        error_lmmse = error_lmmse+1;
    end
    if (nnz(x-z) ~= 0)
        error_opt = error_opt+1;
    end
end