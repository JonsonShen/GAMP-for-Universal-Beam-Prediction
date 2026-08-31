%% run_knownSetB_genieLMMSE_vs_GAMP_singlefile_Gh.m
% Self-contained: only needs CDL_A_modified.csv (or CDL_A.csv)
%
% Goal:
% - Known setB (8 wide beams) observed.
% - Baseline: Genie-LMMSE predicts setA from setB using training mean/cov.
% - Proposed: Sum-product GAMP estimates latent h (independent exponentials), with Exp noise output channel,
%            then predicts y_A = G_A h and ranks [y_A_pred ; y_B_obs].
%
% Also plots:
% - Fig1: Top-1 error rate (argmax over all 40 beams)
% - Fig2: setA prediction MSE (per element) comparing predicted setA vs true noisy setA
%
% Model (notation aligned with pseudo code / paper):
%   h_j ~ Exp(rate=lambda_j),  h_j >= 0
%   z = G h
%   y = z + w,  w ~ Exp(rate=psi), w>=0
%   psi set from target SNR via E[w^2]=2/psi^2 = P_noise
%
% Notes:
% - MATLAB supports scripts with local functions (R2016b+). [web:139]

clear; clc; close all;
rng(1,'twister');

%% ===================== User settings =====================
% csv_file = 'CDL_A.csv';
% n_cluster = 21;
csv_file = 'CDL_A_modified.csv';
n_cluster = 12;

snr_dB_training = 25;
snr_dB_list = -10:5:25;

n_train = 200;     % LMMSE training samples
n_test  = 5000;    % per SNR

n_beams_w = 8;
n_beams_n = 32;
n_beams   = n_beams_w + n_beams_n;  % 40

k2 = n_beams_w;        % observed setB beams
k1 = n_beams - k2;     % setA beams (unobserved)
obs_indices = (k1+1):n_beams;       % last 8 beams are wide

use_column_normalization = true;

% ----- GAMP params -----
gamp.max_iter = 200;
gamp.tol = 1e-6;
gamp.damp = 0.6;
gamp.tau_min = 1e-12;
gamp.tau_max = 1e12;

%% ===================== Load CDL =====================
assert(exist(csv_file,'file')==2, 'CSV not found: %s', csv_file);
CDL_h = load_CDL_local(csv_file, n_cluster);  % [power_linear, AoD, AoA, ZoD, ZoA]
AoD = CDL_h(:,2);  ZoD = CDL_h(:,4);
AoA = CDL_h(:,3);  ZoA = CDL_h(:,5);

%% ===================== Beam setup =====================
% Tx beams
range_aop = 60; range_zop = 30; range_aop_fine = 30;
aop_tx = 0; zop_tx = 90;

tx_params_n = generate_tx_para_local(8,4, 8,4, range_aop_fine, range_zop, aop_tx, zop_tx); % 32 narrow
tx_params_w = generate_tx_para_local(4,2, 4,2, range_aop,      range_zop, aop_tx, zop_tx); % 8 wide
tx_params = [tx_params_n; tx_params_w];  % 40 beams total

% Rx beam
M_rx=2; N_rx=2; aop_rx=-180; zop_rx=90;
rx_params = generate_tx_para_local(M_rx, N_rx, 1, 1, 0, 0, aop_rx, zop_rx);

%% ===================== Build Genie G (40 x n_cluster) =====================
g_rx = BM_gain_local(rx_params(1,:), ZoA, AoA);
g_rx = g_rx(:);

G_all = zeros(n_beams, n_cluster);
for b = 1:n_beams
    g_tx = BM_gain_local(tx_params(b,:), ZoD, AoD);
    g_tx = g_tx(:);
    G_all(b,:) = (g_tx .* g_rx).';
end
G_obs = G_all(obs_indices,:);

%% ===================== Prior for latent h: Exp(mean=2*sigma^2) =====================
sigma = CDL_h(:,1);
mu_h = 2*(sigma.^2);                  % mean of h
lambda = 1 ./ max(mu_h, 1e-12);       % rate

%% ===================== Column normalization =====================
% We run GAMP on h' = d .* h with G_use = G ./ d' so that z = G h = G_use h' unchanged.
% For Exp prior: lambda' = lambda / d.
if use_column_normalization
    col_norms = sqrt(sum(G_obs.^2,1)).';
    col_norms(col_norms < 1e-12) = 1;
    G_obs_use = G_obs ./ (col_norms.');
    G_all_use = G_all ./ (col_norms.');
    lambda_use = lambda ./ col_norms;
else
    G_obs_use = G_obs;
    G_all_use = G_all;
    lambda_use = lambda;
end

%% ===================== Train genie-LMMSE (known setB) =====================
% Y_train: n_train x 40
[Y_train, ~] = generate_y_vectorized(G_all, mu_h, snr_dB_training, n_train);

y_mean = mean(Y_train,1).';
y_cov  = cov(Y_train);

mu_A = y_mean(1:k1);
mu_B = y_mean(obs_indices);

Sigma_AB = y_cov(1:k1, obs_indices);
Sigma_BB = y_cov(obs_indices, obs_indices);

LMMSE_mat = Sigma_AB / (Sigma_BB + 1e-10*eye(k2));

%% ===================== Evaluate =====================
err_lmmse  = zeros(length(snr_dB_list),1);
err_gamp   = zeros(length(snr_dB_list),1);
mseA_lmmse = zeros(length(snr_dB_list),1);
mseA_gamp  = zeros(length(snr_dB_list),1);

for i_snr = 1:length(snr_dB_list)
    snr_dB = snr_dB_list(i_snr);

    % Y_test: n_test x 40, psi_vec: n_test x 1
    [Y_test, psi_vec] = generate_y_vectorized(G_all, mu_h, snr_dB, n_test);

    cnt_l = 0;
    cnt_g = 0;
    seA_l = 0;   % sum squared error on setA
    seA_g = 0;

    for t = 1:n_test
        y_full   = Y_test(t,:).';
        y_A_true = y_full(1:k1);
        y_B_obs  = y_full(obs_indices);

        psi = psi_vec(t);

        % GT top-1 index from full noisy y
        [~, idx_gt] = max(y_full);

        %% ---- Genie-LMMSE ----
        yA_hat_lmmse = mu_A + LMMSE_mat * (y_B_obs - mu_B);

        y_hat_lmmse = -1e9*ones(n_beams,1);
        y_hat_lmmse(1:k1) = yA_hat_lmmse;
        y_hat_lmmse(obs_indices) = y_B_obs;

        [~, idx_l] = max(y_hat_lmmse);
        cnt_l = cnt_l + (idx_l ~= idx_gt);

        seA_l = seA_l + norm(yA_hat_lmmse - y_A_true)^2;

        %% ---- GAMP (estimate latent h from y_B) ----
        h_hat = gamp_expprior_expnoise(y_B_obs, psi, G_obs_use, lambda_use, gamp);

        % Predict set A and add mean noise E[w]=1/psi
        yA_hat_gamp = (G_all_use(1:k1,:) * h_hat) + 1/psi;

        y_hat_gamp = -1e9*ones(n_beams,1);
        y_hat_gamp(1:k1) = yA_hat_gamp;
        y_hat_gamp(obs_indices) = y_B_obs;

        [~, idx_g] = max(y_hat_gamp);
        cnt_g = cnt_g + (idx_g ~= idx_gt);

        seA_g = seA_g + norm(yA_hat_gamp - y_A_true)^2;
    end

    err_lmmse(i_snr) = cnt_l / n_test;
    err_gamp(i_snr)  = cnt_g / n_test;

    mseA_lmmse(i_snr) = seA_l / (n_test * k1);
    mseA_gamp(i_snr)  = seA_g / (n_test * k1);

    fprintf('SNR=%5.1f dB | Top1 err: GAMP=%.4f LMMSE=%.4f | setA MSE: GAMP=%.3e LMMSE=%.3e\n', ...
        snr_dB, err_gamp(i_snr), err_lmmse(i_snr), mseA_gamp(i_snr), mseA_lmmse(i_snr));
end

%% ===================== Plot 1: Top-1 error =====================
figure;
semilogy(snr_dB_list, err_gamp, '-x', 'LineWidth', 2); hold on;
semilogy(snr_dB_list, err_lmmse, '-o', 'LineWidth', 2);
grid on; xlabel('SNR(dB)'); ylabel('Top-1 error rate');
legend('GAMP (Exp prior + Exp noise)', 'Genie-LMMSE (known setB)', 'Location','best');
title('Known setB (8 wide beams): Top-1 error');

%% ===================== Plot 2: setA prediction MSE =====================
figure;
semilogy(snr_dB_list, mseA_gamp, '-x', 'LineWidth', 2); hold on;
semilogy(snr_dB_list, mseA_lmmse, '-o', 'LineWidth', 2);
grid on; xlabel('SNR(dB)'); ylabel('setA prediction MSE (per element)');
legend('GAMP (Exp prior + Exp noise)', 'Genie-LMMSE (known setB)', 'Location','best');
title('Known setB: setA prediction MSE');

%% =====================================================================
%% =========================== Local functions ==========================
%% =====================================================================

function CDL_h = load_CDL_local(csv_file, n_cluster)
    data = readmatrix(csv_file);
    data = data(~any(isnan(data),2),:);
    % Assume columns: [Cluster#, Delay, Power(dB), AoD, AoA, ZoD, ZoA]
    power_dB = data(1:n_cluster, 3);
    AoD = data(1:n_cluster, 4);
    AoA = data(1:n_cluster, 5);
    ZoD = data(1:n_cluster, 6);
    ZoA = data(1:n_cluster, 7);
    AoA = sign(AoA).*(180-abs(AoA));   % panel rotation used in earlier scripts
    power_linear = 10 .^ (power_dB / 20);
    CDL_h = [power_linear, AoD, AoA, ZoD, ZoA];
end

function x = generate_tx_para_local(M, N, n_aop, n_zop, range_aop, range_zop, center_aop, center_zop)
    n_beams = n_aop*n_zop;
    temp = linspace(center_aop-range_aop, center_aop+range_aop, 2*n_aop+1);
    aop = temp(2:2:end);
    temp = linspace(center_zop-range_zop, center_zop+range_zop, 2*n_zop+1);
    zop = temp(2:2:end);
    [temp_x, temp_y] = meshgrid(aop, zop);
    x = [M*ones(n_beams,1), N*ones(n_beams,1), temp_x(:), temp_y(:)];
end

function gain = BM_gain_local(params, Z, A)
    M = params(1);
    N = params(2);
    AoPeak = params(3);
    ZoPeak = params(4);

    sep = 1/2;
    ppM_deg = sep * 360 * sind(AoPeak);
    ppN_deg = sep * 360 * cosd(ZoPeak) / sqrt(1 - (sind(ZoPeak))^2 * (sind(AoPeak))^2);

    phi_3dB   = 90;
    theta_3dB = 90;
    A_m = 30;
    SLA_v = 30;
    G_E_max = 5.5;

    Gain_element = Gain_per_element_local(phi_3dB, theta_3dB, A_m, SLA_v, G_E_max, Z, A);
    gain = Gain_BM_local(M, N, Gain_element, ppM_deg, ppN_deg, Z, A) / (M*N);
end

function x = Gain_per_element_local(phi_3dB, theta_3dB, A_m, SLA_v, G_E_max, theta, phi)
    x = zeros(length(theta),1);
    for iloop = 1:length(theta)
        theta_tmp = theta(iloop);
        A_E_V = -min(12*((theta_tmp-90)/theta_3dB)^2, SLA_v);
        phi_tmp = phi(iloop);
        A_E_H = -min(12*((phi_tmp)/phi_3dB)^2, A_m);
        x(iloop) = G_E_max - min(A_m, -(A_E_H + A_E_V));
    end
end

function x = Gain_BM_local(M, N, Gain_element, ppM_deg, ppN_deg, theta, phi)
    sep = 1/2;
    ppM = ppM_deg*pi/180;
    ppN = ppN_deg*pi/180;

    FieldStrength_lin = zeros(size(Gain_element));
    FieldStrength_element_lin = 10.^(Gain_element/20);

    for n = 1:N
        for m = 1:M
            el_location = [0, (m-1)*sep, (n-1)*sep];
            for iloop = 1:length(theta)
                theta_tmp = theta(iloop)*pi/180;
                phi_tmp = phi(iloop)*pi/180;
                r_hat = [cos(phi_tmp)*sin(theta_tmp), ...
                         sin(phi_tmp)*sin(theta_tmp), ...
                         cos(theta_tmp)];
                ph_adv = 2*pi*(dot(r_hat, el_location));
                net_phase = ph_adv - (m-1)*ppM - (n-1)*ppN;
                FieldStrength_lin(iloop) = FieldStrength_lin(iloop) + ...
                    FieldStrength_element_lin(iloop)*exp(1i*net_phase);
            end
        end
    end
    x = abs(FieldStrength_lin).^2;
end

function [Y, psi_vec] = generate_y_vectorized(G_all, mu_h, snr_dB, n_samp)
    % Vectorized generation:
    % H: (n_cluster x n_samp), Z: (n_beams x n_samp), W: (n_beams x n_samp)
    % w ~ Exp(mean=1/psi), psi set from SNR
    [n_beams, ~] = size(G_all);

    % latent exponentials: h_j ~ Exp(mean=mu_h(j))
    H = exprnd(repmat(mu_h, 1, n_samp));          % (n_cluster x n_samp)

    % noiseless outputs
    Z = G_all * H;                                % (n_beams x n_samp)

    % SNR-based psi
    snr_lin = 10^(snr_dB/10);
    P_sig = mean(Z.^2, 1);                        % (1 x n_samp)
    P_sig(P_sig < 1e-12) = 1e-12;
    P_noise = P_sig / snr_lin;

    % E[w^2] = 2/psi^2 = P_noise  => psi = sqrt(2/P_noise)
    psi_vec = sqrt(2 ./ max(P_noise, 1e-12));     % (1 x n_samp)

    % noise: w ~ Exp(rate=psi), mean=1/psi
    W = exprnd(1 ./ repmat(psi_vec, n_beams, 1));

    Y = (Z + W).';                                % (n_samp x n_beams)
    psi_vec = psi_vec(:);                         % (n_samp x 1)
end

%% ===================== GAMP core (Exp prior + Exp noise) =====================
function h_hat = gamp_expprior_expnoise(y, psi, G, lambda, prm)
    % Sum-product GAMP for:
    %   y = G h + w,  w ~ Exp(rate=psi), w>=0
    %   h_j ~ Exp(rate=lambda_j), h>=0

    [M,~] = size(G);
    damp     = prm.damp;
    max_iter = prm.max_iter;
    tol      = prm.tol;
    tau_min  = prm.tau_min;
    tau_max  = prm.tau_max;

    % Initialize \hat{h}^{(0)} = E[h] = 1/lambda, tau^h = Var(h)=1/lambda^2
    h_hat = 1 ./ lambda;
    tau_h = 1 ./ (lambda.^2);

    s_hat = zeros(M,1);

    G2 = G.^2;

    for it = 1:max_iter
        h_hat_old = h_hat;

        % ---- Output linear step ----
        tau_p = G2 * tau_h;
        tau_p = min(max(tau_p, tau_min), tau_max);

        p_hat = G * h_hat - tau_p .* s_hat;

        % ---- Output nonlinear step ----
        [z_hat, tau_z] = post_z_expnoise(y, p_hat, tau_p, psi);
        tau_z = min(max(tau_z, tau_min), tau_p);

        s_hat_new = (z_hat - p_hat) ./ tau_p;

        tau_s = (1 ./ tau_p) .* max(1 - tau_z ./ tau_p, 0);
        tau_s = min(max(tau_s, tau_min), tau_max);

        % ---- Input linear step ----
        denom = G2.' * tau_s;
        denom = max(denom, tau_min);

        tau_r = 1 ./ denom;
        tau_r = min(max(tau_r, tau_min), tau_max);

        r_hat = h_hat + tau_r .* (G.' * s_hat_new);

        % ---- Input nonlinear step ----
        [h_hat_new, tau_h_new] = post_h_expprior(r_hat, tau_r, lambda);
        tau_h_new = min(max(tau_h_new, tau_min), tau_max);

        % ---- Damping ----
        h_hat = damp*h_hat_new + (1-damp)*h_hat;
        tau_h = damp*tau_h_new + (1-damp)*tau_h;
        s_hat = damp*s_hat_new + (1-damp)*s_hat;

        if norm(h_hat - h_hat_old)/(norm(h_hat_old)+1e-9) < tol
            break;
        end
    end

    h_hat = max(real(h_hat), 0);
end

function [h_post, tau_h_post] = post_h_expprior(r_hat, tau_r, lambda)
    % Posterior for h>=0 with Exp(rate=lambda) prior and Gaussian pseudo-measurement:
    %   r_hat = h + N(0,tau_r)
    % p(h|r_hat) ∝ exp(-lambda*h) * N(h;r_hat,tau_r) * 1(h>=0)
    %
    % Equivalent to lower-truncated normal with shifted mean:
    %   mu_eff = r_hat - lambda*tau_r
    % h(alpha) = phi(alpha)/Phi(-alpha) computed stably by erfcx.

    tau_r = max(real(tau_r), 1e-12);
    r_hat = real(r_hat);

    mu_eff = r_hat - lambda .* tau_r;
    sigma  = sqrt(tau_r);

    alpha = -mu_eff ./ sigma;  % lower truncation at 0
    u = alpha ./ sqrt(2);

    % h(alpha) = phi(alpha)/Phi(-alpha) = sqrt(2/pi)/erfcx(alpha/sqrt(2))
    hcorr = sqrt(2/pi) ./ max(erfcx(u), 1e-300);

    h_post = mu_eff + sigma .* hcorr;
    tau_h_post = tau_r .* (1 + alpha .* hcorr - hcorr.^2);

    h_post = max(h_post, 0);
    tau_h_post = max(tau_h_post, 1e-12);
end

function [z_post, tau_z_post] = post_z_expnoise(y, p_hat, tau_p, psi)
    % Output channel:
    %   y = z + w,  w~Exp(rate=psi), w>=0  =>  z <= y
    % posterior ∝ N(z; p_hat, tau_p) * exp(psi*z) * 1(z<=y)
    % => upper-truncated normal with shifted mean mu_z = p_hat + psi*tau_p

    tau_p = max(real(tau_p), 1e-12);
    p_hat = real(p_hat);
    y     = real(y);

    mu_z  = p_hat + psi .* tau_p;
    sigma = sqrt(tau_p);

    beta = (y - mu_z) ./ sigma;   % upper truncation point
    u = -beta ./ sqrt(2);

    % rho(beta)=phi(beta)/Phi(beta) = sqrt(2/pi)/erfcx(-beta/sqrt(2))
    rho = sqrt(2/pi) ./ max(erfcx(u), 1e-300);

    z_post = mu_z - sigma .* rho;
    tau_z_post = tau_p .* (1 - rho .* (beta + rho));
    tau_z_post = max(tau_z_post, 1e-12);
end
