snr_dB_list = linspace(-10,10,5);
snr_dB_training = 10;
error_counter = zeros(length(snr_dB_list),1);
error_counter_sample = error_counter;



n_beams_w = 4;
n_beams_n = 32; %number of beams
n_beams = n_beams_w + n_beams_n;
k_max = 1; %number of max beams reported to BS
n_samples = 1000; % number of samples used in training
n_test = 1000;
%n_AoA = 5;
%n_AoD = 5;
n_cluster = 12; %max 21 for CDL_A.csv
%CDL_x = zeros(n_cluster,5); %power, AoD, AoA, ZoD, ZoA
k2 = n_beams_w; %set B size
k1 = n_beams - k2; %set A size

% Load the CDL matrix from the CSV file
% CDL_x = [Power (linear), AoD, AoA, ZoD, ZoA]
csv_file = 'CDL_A_modified.csv';
CDL_x = load_CDL(csv_file,n_cluster);

range_aop = 60;
range_zop = 30;
aop_tx = 0;
zop_tx = 90;


M = 8;
N = 4;
n_aop = 8;
n_zop = 4;
tx_params_n = generate_tx_para(M, N,n_aop, n_zop, range_aop, range_zop,aop_tx,zop_tx);

M = 4;
N = 2;
n_aop = 4;
n_zop = 2;
tx_params_w = generate_tx_para(M, N,n_aop, n_zop, range_aop, range_zop,aop_tx,zop_tx);

tx_params = [tx_params_n; tx_params_w];


aop_rx = 0;
zop_rx = 90;
rx_params = generate_tx_para(2, 2, 1, 1, 0, 0, aop_rx,zop_rx);

%sample mean and variance
%Y_sample: rows are y vectors

Y_sample = generate_y(CDL_x,tx_params,rx_params,n_samples,snr_dB_training);
y_mean = mean(Y_sample);
y_cov = cov(Y_sample);
[y_max_sample, I_max_sample]=maxk(y_mean, k_max);
pred_A_mmse = trace(y_cov(1:k1,1:k1)-y_cov(1:k1,k1+1:end)/y_cov(k1+1:end,k1+1:end)*y_cov(1:k1,k1+1:end).');

for i_snr = 1:length(snr_dB_list)
Y_test = generate_y(CDL_x,tx_params,rx_params,n_test,snr_dB_list(i_snr));

%linear prediction with correct data

%index, Set A: 1 to k1, Set B: k1+1 to end
%error_counter = 0;
error_acc = 0;
%error_counter_sample = 0;
error_acc_sample = 0;

log_prd_I = zeros(n_test,1);

for i_test = 1:n_test
    y_B = Y_test(i_test,k1+1:end);

    pred_A = y_mean(1:k1)'+y_cov(1:k1,k1+1:end)/y_cov(k1+1:end,k1+1:end)*(y_B-y_mean(k1+1:end)).';
    
    [y_max_prd,I_max_prd]=maxk([pred_A;y_B.'], k_max);  % reported max RSRP and beam indexes
    log_prd_I(i_test) = I_max_prd;
    [y_max_gt,I_max_gt]=maxk(Y_test(i_test,:), k_max);
    
    error_counter(i_snr) = error_counter(i_snr)+nnz(I_max_prd-I_max_gt);
    error_acc = error_acc+norm(y_max_prd-y_max_gt);

    error_counter_sample(i_snr) = error_counter_sample(i_snr)+nnz(I_max_sample-I_max_gt);
    error_acc_sample = error_acc_sample+norm(y_max_sample-y_max_gt);    
end


end

%Rx gain detection



