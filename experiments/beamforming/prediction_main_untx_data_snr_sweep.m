%snr_dB_training = 10;
snr_dB_list = linspace(-10,10,11);
error_counter = zeros(length(snr_dB_list),1);
error_counter_genie = error_counter;
error_counter_sample = error_counter;
error_counter_rx = error_counter;

n_beams_w = 8;
n_beams_n = 32; %number of beams
n_beams = n_beams_w + n_beams_n;
k_max = 2; %number of max beams reported to BS
n_samples = 2000; % number of samples used in training
n_test = 30000;
%n_AoA = 5;
%n_AoD = 5;

%CDL_x = zeros(n_cluster,5); %power, AoD, AoA, ZoD, ZoA
k2 = n_beams_w; %set B size
k1 = n_beams - k2; %set A size

% Load the CDL matrix from the CSV file
% CDL_x = [Power (linear), AoD, AoA, ZoD, ZoA]
n_cluster = 20;
csv_file = 'CDL_A.csv';
CDL_x = load_CDL(csv_file,n_cluster);

%Rx side Tx parameter assumptions
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

tx_params_rx_asp = [tx_params_n; tx_params_w];


%Actual Tx parameters
range_aop = 60;
range_zop = 30;
range_aop_fine = 30;

aop_tx = 0;
zop_tx = 90;


M = 8;
N = 4;
n_aop = 8;
n_zop = 4;
tx_params_n = generate_tx_para(M, N,n_aop, n_zop, range_aop_fine, range_zop,aop_tx,zop_tx);

M = 4;
N = 2;
n_aop = 4;
n_zop = 2;
tx_params_w = generate_tx_para(M, N,n_aop, n_zop, range_aop, range_zop,aop_tx,zop_tx);

tx_params = [tx_params_n; tx_params_w];

M_rx = 2;
N_rx = 2;
aop_rx = -180;
zop_rx = 90;
rx_params = generate_tx_para(M_rx, M_rx, 1, 1, 0, 0, aop_rx,zop_rx);

%Rx side training
Y_sample_rx = zeros(n_samples*length(snr_dB_list), length(tx_params(:,1)));
Y_sample_rx_genie = zeros(n_samples*length(snr_dB_list), length(tx_params(:,1)));
for i_snr = 1:length(snr_dB_list)
	Y_sample_rx_snr = generate_y(CDL_x,tx_params_rx_asp,rx_params,n_samples,snr_dB_list(i_snr));
	Y_sample_rx_genie_snr = generate_y(CDL_x,tx_params,rx_params,n_samples,snr_dB_list(i_snr));
	Y_sample_rx(n_samples*(i_snr-1)+1:n_samples*(i_snr-1)+n_samples,:) = Y_sample_rx_snr;
	Y_sample_rx_genie(n_samples*(i_snr-1)+1:n_samples*(i_snr-1)+n_samples,:) = Y_sample_rx_genie_snr;
end

y_mean_rx = mean(Y_sample_rx);
y_cov_rx = cov(Y_sample_rx);

y_mean_rx_genie = mean(Y_sample_rx_genie);
y_cov_rx_genie = cov(Y_sample_rx_genie);
[y_max_sample, I_max_sample]=maxk(y_mean_rx_genie, k_max);

%Tx side training for Rx gain detection: use Rx gain hypothesis and Rx side
%assumption for Tx BM parameters

aop_rx = 0;
zop_rx = 90;
range_aop = 60;
range_zop = 30;
n_aop = 9;
n_zop = 5;
n_rxg_hypo = n_aop*n_zop;
rx_params_hypo = generate_tx_para(M_rx, N_rx,n_aop, n_zop, range_aop, range_zop,aop_rx,zop_rx);

Y_sample_Rxg = zeros(n_rxg_hypo,n_samples*length(snr_dB_list),n_beams);
Y_sample_tx = zeros(n_rxg_hypo,n_samples*length(snr_dB_list),n_beams);
y_mean_Rxg = zeros(n_rxg_hypo,n_beams);
y_cov_Rxg = zeros(n_rxg_hypo,n_beams,n_beams);
y_mean_tx = zeros(n_rxg_hypo,n_beams);
y_cov_tx = zeros(n_rxg_hypo,n_beams,n_beams);

for i_rx_hypo = 1:n_rxg_hypo
	Y_sample_Rxg_temp = zeros(n_samples*length(snr_dB_list), length(tx_params(:,1)));
	Y_sample_tx_temp = zeros(n_samples*length(snr_dB_list), length(tx_params(:,1)));
	for i_snr = 1:length(snr_dB_list)
		Y_sample_Rxg_snr = generate_y(CDL_x,tx_params_rx_asp,rx_params_hypo(i_rx_hypo,:),n_samples,snr_dB_list(i_snr));
		Y_sample_tx_snr = generate_y(CDL_x,tx_params,rx_params_hypo(i_rx_hypo,:),n_samples,snr_dB_list(i_snr));
		Y_sample_Rxg_temp(n_samples*(i_snr-1)+1:n_samples*(i_snr-1)+n_samples,:) = Y_sample_Rxg_snr;
		Y_sample_tx_temp(n_samples*(i_snr-1)+1:n_samples*(i_snr-1)+n_samples,:) = Y_sample_tx_snr;
	end
	Y_sample_Rxg(i_rx_hypo,:,:) = Y_sample_Rxg_temp;
	Y_sample_tx(i_rx_hypo,:,:) = Y_sample_tx_temp;
    y_mean_Rxg(i_rx_hypo,:) = mean(squeeze(Y_sample_Rxg(i_rx_hypo,:,:)));
    y_cov_Rxg(i_rx_hypo,:,:) = cov(squeeze(Y_sample_Rxg(i_rx_hypo,:,:)));
    y_mean_tx(i_rx_hypo,:) = mean(squeeze(Y_sample_tx(i_rx_hypo,:,:)));
    y_cov_tx(i_rx_hypo,:,:) = cov(squeeze(Y_sample_tx(i_rx_hypo,:,:)));    
end



%inference
for i_snr = 1:length(snr_dB_list)
Y_test = generate_y(CDL_x,tx_params,rx_params,n_test,snr_dB_list(i_snr));

diff_y_rxg = zeros(n_rxg_hypo,1);

log_prd_I = zeros(n_test,k_max);
log_prd_I_tx = zeros(n_test,k_max);
log_prd_I_gt = zeros(n_test,k_max);

% error_counter = 0;
% error_acc = 0;
% error_counter_sample = 0;
% error_acc_sample = 0;

for i_test = 1:n_test
    %rx side inference
    y_B = Y_test(i_test,k1+1:end);

    pred_A_rx = y_mean_rx(1:k1)'+y_cov_rx(1:k1,k1+1:end)/y_cov_rx(k1+1:end,k1+1:end)*(y_B-y_mean_rx(k1+1:end)).';
    pred_A_rx_genie = y_mean_rx_genie(1:k1)'+y_cov_rx_genie(1:k1,k1+1:end)/y_cov_rx_genie(k1+1:end,k1+1:end)*(y_B-y_mean_rx_genie(k1+1:end)).';
    
    [y_max_prd_rx,I_max_prd_rx]=maxk([pred_A_rx;y_B.'], k_max);  % reported max RSRP and beam indexes
    log_prd_I(i_test,:) = I_max_prd_rx;
    [y_max_prd_rx_genie,I_max_prd_rx_genie]=maxk([pred_A_rx_genie;y_B.'], k_max);

    [y_max_gt,I_max_gt]=maxk(Y_test(i_test,:), k_max);
    log_prd_I_gt(i_test,:) = I_max_gt;
    
    %tx side post-processing
    for i_rx_hypo = 1:n_rxg_hypo
        pred_A_rxg = y_mean_Rxg(i_rx_hypo,1:k1).'+squeeze(y_cov_Rxg(i_rx_hypo,1:k1,k1+1:end))/squeeze(y_cov_Rxg(i_rx_hypo,k1+1:end,k1+1:end))*squeeze(y_B-y_mean_Rxg(i_rx_hypo,k1+1:end)).';
        y_all_rxg = [pred_A_rxg; y_B.'];
        diff_y_rxg(i_rx_hypo)=norm(y_all_rxg(I_max_prd_rx)-y_max_prd_rx);
    end
    [temp, rxg_max_idx]=min(diff_y_rxg); 
    rx_params_max = rx_params_hypo(rxg_max_idx,:);
    
    %Tx prediction after Rx gain is detected
    pred_A_tx = y_mean_tx(rxg_max_idx,1:k1).'+squeeze(y_cov_tx(rxg_max_idx,1:k1,k1+1:end))/squeeze(y_cov_tx(rxg_max_idx,k1+1:end,k1+1:end))*squeeze(y_B-y_mean_tx(rxg_max_idx,k1+1:end)).';

    [y_max_prd_tx,I_max_prd_tx]=maxk([pred_A_tx;y_B.'], k_max);  % reported max RSRP and beam indexes
    log_prd_I_tx(i_test,:) = I_max_prd_tx;

    error_counter(i_snr) = error_counter(i_snr)+nnz(I_max_prd_tx(1)-I_max_gt(1));
    %error_acc = error_acc+norm(y_max_prd_tx-y_max_gt);

    error_counter_sample(i_snr) = error_counter_sample(i_snr)+nnz(I_max_sample(1)-I_max_gt(1));
    %error_acc_sample = error_acc_sample+norm(y_max_sample-y_max_gt);

    error_counter_genie(i_snr) = error_counter_genie(i_snr)+nnz(I_max_prd_rx_genie(1)-I_max_gt(1));
    error_counter_rx(i_snr) = error_counter_rx(i_snr)+nnz(I_max_prd_rx(1)-I_max_gt(1));


end %i_test
end %i_snr
figure;
semilogy(snr_dB_list,error_counter_genie/n_test,'-o',snr_dB_list,error_counter/n_test,'-hex',snr_dB_list,error_counter_rx/n_test,'-*',snr_dB_list,error_counter_sample/n_test,'-+');
xlabel('SNR(dB)');ylabel('Prediction error rate');grid on;grid minor;legend('Linear prediction known CB','Universal scheme','Linear prediction unknown CB','Legacy scheme');title('Prediction error rate comparison');





