n_beams = 64; %number of beams
k_max = 2; %number of max beams reported to BS
n_samples = 1000; % number of samples used in training
%n_AoA = 5;
%n_AoD = 5;
n_cluster = 5;
CDL_x = zeros(n_cluster,5); %power, AoD, AoA, ZoD, ZoA
k1 = 54; %set A size


%sample mean and variance
%Y_sample: rows are y vectors
Y_sample = zeros(n_samples,n_beams);
for i_tx = 1:n_beams
    %Y_sample(:,i_tx) = generate_y(CDL_x,tx_params(i_tx,:),rx_params,n_samples);
end
y_mean = mean(Y_sample);
y_cov = cov(Y_sample);

%index, Set A: 1 to k1, Set B: k1+1 to end


%linear prediction with correct data
pred_A = y_mean(1:k1)+y_cov(1:k1,k1+1:end)\(y_cov(k1+1:end,k1+1:end))*(y_B-y_mean(k1+1:end));
pred_A_mmse = trace(y_cov(1:k1,1:k1)-y_cov(1:k1,k1+1:end)\(y_cov(k1+1:end,k1+1:end))*y_cov(1:k1,k1+1:end)');


[y_max,I_max]=maxk([pred_A y_B], k_max); % reported max RSRP and beam indexes


%Rx gain detection
n_rxg_hypo = 64;
rxg_hypo = zeros(n_rxg_hypo,n_cluster);
Y_sample_Rxg = zeros(n_rxg_hypo,n_samples,n_beams);
diff_y_rxg = zeros(64,1);
for rxg_i = 1:n_rxg_hypo
    y_mean_Rxg = mean(Y_sample_Rxg(rxg_i,:,:));
    y_cov_Rxg = cov(Y_sample_Rxg(rxg_i,:,:));
    pred_A_rxg = y_mean_Rxg(1:k1)+y_cov_Rxg(1:k1,k1+1:end)\(y_cov_Rxg(k1+1:end,k1+1:end))*(y_B-y_mean_Rxg(k1+1:end));
    y_all_rxg = [pred_A_rxg y_B];
    diff_y_rxg(rxg_i)=norm(y_all_rxg(I_max)-y_max);
end
[temp, rxg_max_idx]=max(diff_y_rxg); 
rxg_max = rxg_hypo(rxg_max_idx,:);

%Tx prediction
Y_sample_tx = zeros(n_samples,n_beams);
y_mean_tx = mean(Y_sample_tx);
y_cov_tx = cov(Y_sample_tx);
pred_A_tx = y_mean_tx(1:k1)+y_cov_tx(1:k1,k1+1:end)\(y_cov_tx(k1+1:end,k1+1:end))*(y_B-y_mean_tx(k1+1:end));

