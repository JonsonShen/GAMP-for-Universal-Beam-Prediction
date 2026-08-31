%% run_knownSetB_genieLMMSE_vs_GAMP_meanRemoval_singlefile.m
% Self-contained: only needs CDL_A.csv
%
% Goal:
% - Known setB (8 wide beams) observed.
% - Baseline: Genie-LMMSE predicts setA from setB using training mean/cov.
% - Proposed: GAMP estimates latent x (independent exponentials), with Exp noise output channel,
%            then predicts y_A = A_A x and ranks [y_A_pred ; y_B_obs].
%
% Adds:
% - mean_remove_A switch in GAMP via matrix augmentation (mean-removal).
% - Fig1: Top-1 error rate over 40 beams
% - Fig2: setA prediction MSE (per element) over 32 narrow beams
%
clear; clc; close all;
rng(1,'twister');

%% ===================== User settings =====================
csv_file = 'CDL_A.csv';
n_cluster = 21;

snr_dB_training = 25;
snr_dB_list = -10:5:25;

n_train = 200;
n_test  = 5000;

n_beams_w = 8;
n_beams_n = 32;
n_beams   = n_beams_w + n_beams_n;  % 40

k2 = n_beams_w;
k1 = n_beams - k2;                  % 32 (setA)
obs_indices = (k1+1):n_beams;        % last 8 are wide beams (setB)

use_column_normalization = true;

% ----- GAMP params -----
gamp.max_iter = 200;
gamp.tol = 1e-6;
gamp.damp = 0.6;          % 0.4~0.9 typically
gamp.tau_min = 1e-12;
gamp.tau_max = 1e12;

% ----- Mean-removal switch -----
gamp.mean_remove_A = true;   % <<<<<<<<<< toggle here
gamp.sigma2_mr = 1e-14;      % AWGN variance for the two constraint rows (smaller -> harder constraint)

%% ===================== Load CDL =====================
assert(exist(csv_file,'file')==2, 'CSV not found: %s', csv_file);
CDL_x = load_CDL_local(csv_file, n_cluster);  % [power_linear, AoD, AoA, ZoD, ZoA]
AoD = CDL_x(:,2);  ZoD = CDL_x(:,4);
AoA = CDL_x(:,3);  ZoA = CDL_x(:,5);

%% ===================== Beam setup =====================
range_aop = 60; range_zop = 30; range_aop_fine = 30;
aop_tx = 0; zop_tx = 90;

tx_params_n = generate_tx_para_local(8,4, 8,4, range_aop_fine, range_zop, aop_tx, zop_tx); % 32 narrow
tx_params_w = generate_tx_para_local(4,2, 4,2, range_aop,      range_zop, aop_tx, zop_tx); % 8 wide
tx_params = [tx_params_n; tx_params_w];  % 40 beams total

M_rx=2; N_rx=2; aop_rx=-180; zop_rx=90;
rx_params = generate_tx_para_local(M_rx, N_rx, 1, 1, 0, 0, aop_rx, zop_rx);

%% ===================== Build GENIE A (40x21) =====================
g_rx = BM_gain_local(rx_params(1,:), ZoA, AoA);
g_rx = g_rx(:);

A_all = zeros(n_beams, n_cluster);
for b = 1:n_beams
    g_tx = BM_gain_local(tx_params(b,:), ZoD, AoD);
    g_tx = g_tx(:);
    A_all(b,:) = (g_tx .* g_rx).';
end
A_obs = A_all(obs_indices,:);         % 8x21

%% ===================== Prior for latent x: Exp(mean=2*sigma^2) =====================
sigma = CDL_x(:,1);
mu_x = 2*(sigma.^2);
lambda = 1 ./ max(mu_x, 1e-12);

%% ===================== Column normalization =====================
if use_column_normalization
    col_norms = sqrt(sum(A_obs.^2,1)).';
    col_norms(col_norms < 1e-12) = 1;

    A_obs_use = A_obs ./ (col_norms.');
    A_all_use = A_all ./ (col_norms.');

    % x' = d*x  =>  lambda' = lambda/d
    lambda_use = lambda ./ col_norms;
else
    A_obs_use = A_obs;
    A_all_use = A_all;
    lambda_use = lambda;
end

%% ===================== Train genie-LMMSE (known setB) =====================
[Y_train, ~] = generate_y_vectorized(A_all, mu_x, snr_dB_training, n_train);  % n_train x 40
y_mean = mean(Y_train,1).';
y_cov  = cov(Y_train);

mu_A = y_mean(1:k1);
mu_B = y_mean(obs_indices);

Sigma_AB = y_cov(1:k1, obs_indices);
Sigma_BB = y_cov(obs_indices, obs_indices);
LMMSE_mat = Sigma_AB / (Sigma_BB + 1e-10*eye(k2));

%% ===================== Evaluate =====================
err_lmmse = zeros(length(snr_dB_list),1);
err_gamp  = zeros(length(snr_dB_list),1);

mseA_lmmse = zeros(length(snr_dB_list),1);
mseA_gamp  = zeros(length(snr_dB_list),1);

for i_snr = 1:length(snr_dB_list)
    snr_dB = snr_dB_list(i_snr);

    [Y_test, psi_vec] = generate_y_vectorized(A_all, mu_x, snr_dB, n_test); % n_test x 40

    cnt_l = 0; cnt_g = 0;
    seA_l = 0; seA_g = 0;

    for t = 1:n_test
        y_full = Y_test(t,:).';
        y_A_true = y_full(1:k1);
        y_B = y_full(obs_indices);
        psi = psi_vec(t);

        % GT top-1 index from full noisy y (40 beams)
        [~, idx_gt] = max(y_full);

        %% ---- Genie-LMMSE ----
        pred_A = mu_A + LMMSE_mat * (y_B - mu_B);

        y_hat_lmmse = -1e9*ones(n_beams,1);
        y_hat_lmmse(1:k1) = pred_A;
        y_hat_lmmse(obs_indices) = y_B;

        [~, idx_l] = max(y_hat_lmmse);
        cnt_l = cnt_l + (idx_l ~= idx_gt);
        seA_l = seA_l + norm(pred_A - y_A_true)^2;

        %% ---- GAMP ----
        x_hat = gamp_expprior_expnoise_mr(y_B, psi, A_obs_use, lambda_use, gamp);

        predA_g = (A_all_use(1:k1,:) * x_hat) + 1/psi;

        y_hat_gamp = -1e9*ones(n_beams,1);
        y_hat_gamp(1:k1) = predA_g;
        y_hat_gamp(obs_indices) = y_B;

        [~, idx_g] = max(y_hat_gamp);
        cnt_g = cnt_g + (idx_g ~= idx_gt);
        seA_g = seA_g + norm(predA_g - y_A_true)^2;
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
grid on; xlabel('SNR(dB)'); ylabel('Top-1 error rate (40 beams)');
lg1 = 'GAMP';
if gamp.mean_remove_A, lg1 = 'GAMP + mean-removal'; end
legend(lg1, 'Genie-LMMSE (known setB)', 'Location','best');
title('Known setB: Top-1 error');

%% ===================== Plot 2: setA prediction MSE =====================
figure;
semilogy(snr_dB_list, mseA_gamp, '-x', 'LineWidth', 2); hold on;
semilogy(snr_dB_list, mseA_lmmse, '-o', 'LineWidth', 2);
grid on; xlabel('SNR(dB)'); ylabel('setA prediction MSE (per element)');
legend(lg1, 'Genie-LMMSE (known setB)', 'Location','best');
title('Known setB: setA prediction MSE');

%% =====================================================================
%% =========================== Local functions ==========================
%% =====================================================================

function CDL_x = load_CDL_local(csv_file, n_cluster)
    data = readmatrix(csv_file);
    data = data(~any(isnan(data),2),:);

    power_dB = data(1:n_cluster, 3);
    AoD = data(1:n_cluster, 4);
    AoA = data(1:n_cluster, 5);
    ZoD = data(1:n_cluster, 6);
    ZoA = data(1:n_cluster, 7);

    AoA = sign(AoA).*(180-abs(AoA));
    power_linear = 10 .^ (power_dB / 20);

    CDL_x = [power_linear, AoD, AoA, ZoD, ZoA];
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

function [Y, psi_vec] = generate_y_vectorized(A_all, mu_x, snr_dB, n_samp)
    [n_beams, ~] = size(A_all);

    X = exprnd(repmat(mu_x,1,n_samp));              % latent exponentials
    Z = A_all * X;

    snr_lin = 10^(snr_dB/10);
    P_sig = mean(Z.^2, 1);
    P_sig(P_sig < 1e-12) = 1e-12;

    P_noise = P_sig / snr_lin;
    psi_vec = sqrt(2 ./ max(P_noise, 1e-12));      % E[w^2]=2/psi^2

    W = exprnd(1 ./ repmat(psi_vec, n_beams, 1));  % mean = 1/psi
    Y = (Z + W).';

    psi_vec = psi_vec(:);
end

%% ===================== GAMP wrapper (with optional mean removal) =====================
function x_hat = gamp_expprior_expnoise_mr(y, psi, A, lambda, prm)
    if isfield(prm,'mean_remove_A') && prm.mean_remove_A
        [Abar, ybar, Mexp, sigma2_mr, Norig] = mean_remove_augment(A, y, prm);
        xbar = gamp_core_mixed_out(Abar, ybar, psi, lambda, Norig, Mexp, sigma2_mr, prm);
        x_hat = xbar(1:Norig);
    else
        x_hat = gamp_core_plain(A, y, psi, lambda, prm);
    end
end

%% ===================== Mean-removal augmentation =====================
function [Abar, ybar, Mexp, sigma2_mr, Norig] = mean_remove_augment(A, y, prm)
    % Implements the augmented matrix structure described in mean-removal GAMP work:
    % create Ahat with (approximately) zero row/column averages, and augment with 2 rows/cols.

    [M,N] = size(A);
    Norig = N;
    oneM = ones(M,1);
    oneN = ones(N,1);

    mu = mean(A(:));              % global mean
    gamma = (A*oneN)/N;           % row averages (Mx1)
    cbar  = (A.'*oneM)/M;         % column averages (Nx1)
    c = cbar - mu*oneN;           % shifted column averages (Nx1)

    % A = Ahat + gamma*1_N^T + 1_M*c^T, and Ahat has ~zero row/col averages
    Ahat = A - gamma*oneN.' - oneM*c.';

    Af = norm(Ahat,'fro');
    eps0 = 1e-12;

    % simple equalization scalars (avoid huge scaling if gamma/c tiny)
    b12 = Af/(norm(gamma)+eps0);
    b13 = Af/(norm(oneM)+eps0);
    b21 = Af/(norm(oneN)+eps0);
    b31 = Af/(norm(c)+eps0);

    Abar = [Ahat,           b12*gamma,        b13*oneM;
            b21*oneN.',    -b21*b12,         0;
            b31*c.',        0,              -b31*b13];

    ybar = [y; 0; 0];
    Mexp = M;

    if isfield(prm,'sigma2_mr')
        sigma2_mr = prm.sigma2_mr;
    else
        sigma2_mr = 1e-14;
    end
end

%% ===================== GAMP core: plain Exp-noise output =====================
function x_hat = gamp_core_plain(A, y, psi, lambda, prm)
    [M,~] = size(A);

    damp     = prm.damp;
    max_iter = prm.max_iter;
    tol      = prm.tol;
    tau_min  = prm.tau_min;
    tau_max  = prm.tau_max;

    x_hat = 1 ./ lambda;
    x_var = 1 ./ (lambda.^2);
    s_hat = zeros(M,1);

    A2 = A.^2;

    for it = 1:max_iter
        x_hat_old = x_hat;

        tau_p = A2 * x_var;
        tau_p = min(max(tau_p, tau_min), tau_max);
        p_hat = A * x_hat - tau_p .* s_hat;

        [z_hat, z_var] = post_z_expnoise(y, p_hat, tau_p, psi);
        z_var = min(max(z_var, tau_min), tau_p);

        s_hat_new = (z_hat - p_hat) ./ tau_p;

        tau_s = (1 ./ tau_p) .* max(1 - z_var ./ tau_p, 0);
        tau_s = min(max(tau_s, tau_min), tau_max);

        denom = (A2.' * tau_s);
        denom = max(denom, tau_min);
        tau_r = 1 ./ denom;
        tau_r = min(max(tau_r, tau_min), tau_max);

        r_hat = x_hat + tau_r .* (A.' * s_hat_new);

        [x_hat_new, x_var_new] = post_x_expprior(r_hat, tau_r, lambda);
        x_var_new = min(max(x_var_new, tau_min), tau_max);

        x_hat = damp*x_hat_new + (1-damp)*x_hat;
        x_var = damp*x_var_new + (1-damp)*x_var;
        s_hat = damp*s_hat_new + (1-damp)*s_hat;

        if norm(x_hat - x_hat_old)/(norm(x_hat_old)+1e-9) < tol
            break;
        end
    end

    x_hat = max(real(x_hat), 0);
end

%% ===================== GAMP core: mixed output (Exp + 2 AWGN constraints) =====================
function xbar = gamp_core_mixed_out(Abar, ybar, psi, lambda, Norig, Mexp, sigma2_mr, prm)
    [Mbar, Nbar] = size(Abar);
    assert(Nbar == Norig+2, 'Augmented N mismatch');

    damp     = prm.damp;
    max_iter = prm.max_iter;
    tol      = prm.tol;
    tau_min  = prm.tau_min;
    tau_max  = prm.tau_max;

    % init xbar: first Norig from Exp prior, last 2 from 0 (flat prior)
    xbar = zeros(Nbar,1);
    xbar(1:Norig) = 1 ./ lambda;
    x_var = zeros(Nbar,1);
    x_var(1:Norig) = 1 ./ (lambda.^2);
    x_var(Norig+1:end) = 1;    % arbitrary; will be overwritten by flat-prior update

    s_hat = zeros(Mbar,1);
    A2 = Abar.^2;

    for it = 1:max_iter
        x_old = xbar;

        tau_p = A2 * x_var;
        tau_p = min(max(tau_p, tau_min), tau_max);
        p_hat = Abar * xbar - tau_p .* s_hat;

        [z_hat, z_var] = post_z_mixed(ybar, p_hat, tau_p, psi, Mexp, sigma2_mr);
        z_var = min(max(z_var, tau_min), tau_p);

        s_hat_new = (z_hat - p_hat) ./ tau_p;

        tau_s = (1 ./ tau_p) .* max(1 - z_var ./ tau_p, 0);
        tau_s = min(max(tau_s, tau_min), tau_max);

        denom = (A2.' * tau_s);
        denom = max(denom, tau_min);
        tau_r = 1 ./ denom;
        tau_r = min(max(tau_r, tau_min), tau_max);

        r_hat = xbar + tau_r .* (Abar.' * s_hat_new);

        % input nonlinear:
        [x1, v1] = post_x_expprior(r_hat(1:Norig), tau_r(1:Norig), lambda);
        % last 2 are flat prior => posterior = Gaussian N(r_hat, tau_r)
        x2 = r_hat(Norig+1:end);
        v2 = tau_r(Norig+1:end);

        x_new = [x1; x2];
        v_new = [v1; v2];

        xbar = damp*x_new + (1-damp)*xbar;
        x_var = damp*v_new + (1-damp)*x_var;
        s_hat = damp*s_hat_new + (1-damp)*s_hat;

        if norm(xbar - x_old)/(norm(x_old)+1e-9) < tol
            break;
        end
    end

    xbar = real(xbar);
end

%% ===================== Scalar posteriors =====================
function [x_post, v_post] = post_x_expprior(r, v_r, lam)
    v_r = max(real(v_r), 1e-12);
    r   = real(r);

    mu_shifted = r - lam .* v_r;
    sigma      = sqrt(v_r);

    alpha = -mu_shifted ./ sigma;

    % h(alpha) = phi(alpha)/Phi(-alpha) computed stably using erfcx
    u = alpha ./ sqrt(2);
    h = sqrt(2/pi) ./ max(erfcx(u), 1e-300);

    x_post = mu_shifted + sigma .* h;
    v_post = v_r .* (1 + alpha .* h - h.^2);

    x_post = max(x_post, 0);
    v_post = max(v_post, 1e-12);
end

function [z_post, vz_post] = post_z_expnoise(y, p_hat, vp, psi)
    vp    = max(real(vp), 1e-12);
    p_hat = real(p_hat);
    y     = real(y);

    mu_new = p_hat + psi .* vp;
    sigma  = sqrt(vp);

    alpha = (y - mu_new) ./ sigma;

    % rho(alpha) = phi(alpha)/Phi(alpha) computed stably using erfcx
    u   = -alpha ./ sqrt(2);
    rho = sqrt(2/pi) ./ max(erfcx(u), 1e-300);

    z_post  = mu_new - sigma .* rho;
    vz_post = vp .* (1 - rho .* (alpha + rho));
    vz_post = max(vz_post, 1e-12);
end

function [z_post, vz_post] = post_z_awgn(y, p_hat, vp, sigma2)
    vp = max(real(vp), 1e-12);
    sigma2 = max(real(sigma2), 1e-18);

    denom = vp + sigma2;
    z_post = (vp .* y + sigma2 .* p_hat) ./ denom;
    vz_post = (vp .* sigma2) ./ denom;
    vz_post = max(vz_post, 1e-18);
end

function [z_post, vz_post] = post_z_mixed(y, p_hat, vp, psi, Mexp, sigma2_mr)
    z_post = zeros(size(y));
    vz_post = zeros(size(y));

    idxExp = 1:Mexp;
    idxGauss = (Mexp+1):length(y);

    if ~isempty(idxExp)
        [z_post(idxExp), vz_post(idxExp)] = post_z_expnoise(y(idxExp), p_hat(idxExp), vp(idxExp), psi);
    end

    if ~isempty(idxGauss)
        [z_post(idxGauss), vz_post(idxGauss)] = post_z_awgn(y(idxGauss), p_hat(idxGauss), vp(idxGauss), sigma2_mr);
    end
end
