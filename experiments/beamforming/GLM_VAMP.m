%% run_knownSetB_genieLMMSE_vs_GLMVAMP_singlefile.m
% Self-contained: only needs CDL_A.csv
%
% Compare:
%  (1) Genie-LMMSE (known codebook, observe 8 wide beams=setB)
%  (2) GLM-VAMP (VAMP for GLM) with Exp prior on latent x and Exp noise output channel [web:70][web:30]
%
% Model:
%   x_j ~ Exp(rate=lambda_j), independent
%   z = A x
%   y = z + w, w ~ Exp(rate=psi), independent, w>=0
%
% Decide:
%   predict setA beams by z_hat + E[w]=1/psi, and keep observed setB as measured.

clear; clc; close all;
rng(1,'twister');

%% ===================== User settings =====================
csv_file = 'CDL_A.csv';
n_cluster = 21;

snr_dB_training = 25;
snr_dB_list = -10:5:25;

n_train = 200;     % LMMSE training samples
n_test  = 5000;    % per SNR (increase later)

n_beams_w = 8;
n_beams_n = 32;
n_beams   = n_beams_w + n_beams_n;

k2 = n_beams_w;
k1 = n_beams - k2;
obs_indices = (k1+1):n_beams;     % last 8 beams are wide = setB

use_column_normalization = true;

% ----- GLM-VAMP params (Algorithm 2 family) -----
vamp.max_iter = 200;
vamp.tol = 1e-6;
vamp.damp = 0.7;

vamp.alpha_min = 1e-3; vamp.alpha_max = 1-1e-3;
vamp.beta_min  = 1e-3; vamp.beta_max  = 1-1e-3;
vamp.gam_min = 1e-12;  vamp.gam_max = 1e12;
vamp.tau_min = 1e-12;  vamp.tau_max = 1e12;

%% ===================== Load CDL =====================
assert(exist(csv_file,'file')==2, 'CSV not found: %s', csv_file);
CDL_x = load_CDL_local(csv_file, n_cluster);  % [power_linear, AoD, AoA, ZoD, ZoA]
AoD = CDL_x(:,2);  ZoD = CDL_x(:,4);
AoA = CDL_x(:,3);  ZoA = CDL_x(:,5);

%% ===================== Beam setup (same as before) =====================
range_aop = 60; range_zop = 30; range_aop_fine = 30;
aop_tx = 0; zop_tx = 90;

tx_params_n = generate_tx_para_local(8,4, 8,4, range_aop_fine, range_zop, aop_tx, zop_tx); % 32 narrow
tx_params_w = generate_tx_para_local(4,2, 4,2, range_aop,      range_zop, aop_tx, zop_tx); % 8 wide
tx_params = [tx_params_n; tx_params_w];  % 40 total

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
A_obs = A_all(obs_indices,:);   % 8x21

%% ===================== Prior for latent x (Exp) =====================
% Match your earlier convention: x_j mean = 2*sigma_j^2
sigma = CDL_x(:,1);
mu_x = 2*(sigma.^2);             % 21x1
lambda = 1 ./ max(mu_x, 1e-12);  % rate

%% ===================== Column normalization =====================
if use_column_normalization
    col_norms = sqrt(sum(A_obs.^2,1)).';
    col_norms(col_norms < 1e-12) = 1;

    A_obs_use = A_obs ./ (col_norms.');
    A_all_use = A_all ./ (col_norms.');

    lambda_use = lambda ./ col_norms;   % x' = d x => rate' = rate/d
else
    A_obs_use = A_obs;
    A_all_use = A_all;
    lambda_use = lambda;
end

% GLM-VAMP linear module uses Gram on obs-space (8x8), good for M<N.
G = A_obs_use * A_obs_use.';

%% ===================== Train Genie-LMMSE on full y =====================
Y_train = generate_y_vectorized(A_all, mu_x, snr_dB_training, n_train);  % n_train x 40
y_mean = mean(Y_train,1).';
y_cov  = cov(Y_train);

mu_A = y_mean(1:k1);
mu_B = y_mean(obs_indices);

Sigma_AB = y_cov(1:k1, obs_indices);
Sigma_BB = y_cov(obs_indices, obs_indices);
LMMSE_mat = Sigma_AB / (Sigma_BB + 1e-10*eye(k2));

%% ===================== Evaluate =====================
err_lmmse = zeros(length(snr_dB_list),1);
err_vamp  = zeros(length(snr_dB_list),1);

for i_snr = 1:length(snr_dB_list)
    snr_dB = snr_dB_list(i_snr);

    [Y_test, psi_vec] = generate_y_vectorized(A_all, mu_x, snr_dB, n_test);
    cnt_l = 0;
    cnt_v = 0;

    for t = 1:n_test
        y_full = Y_test(t,:).';
        y_B = y_full(obs_indices);
        psi = psi_vec(t);

        [~, idx_gt] = max(y_full);

        % ---- Genie-LMMSE ----
        pred_A = mu_A + LMMSE_mat * (y_B - mu_B);
        y_hat_lmmse = -1e9*ones(n_beams,1);
        y_hat_lmmse(1:k1) = pred_A;
        y_hat_lmmse(obs_indices) = y_B;
        [~, idx_l] = max(y_hat_lmmse);
        cnt_l = cnt_l + (idx_l ~= idx_gt);

        % ---- GLM-VAMP: infer latent x from setB ----
        x_hat = glmvamp_exp_exp_xonly(y_B, psi, lambda_use, A_obs_use, G, vamp);

        % Predict setA: y_A ≈ z_A_hat + E[w]=1/psi, setB uses observed y_B
        y_hat_vamp = -1e9*ones(n_beams,1);
        y_hat_vamp(1:k1) = (A_all_use(1:k1,:) * x_hat) + 1/psi;
        y_hat_vamp(obs_indices) = y_B;

        [~, idx_v] = max(y_hat_vamp);
        cnt_v = cnt_v + (idx_v ~= idx_gt);
    end

    err_lmmse(i_snr) = cnt_l / n_test;
    err_vamp(i_snr)  = cnt_v / n_test;
    fprintf('SNR=%5.1f dB | VAMP err=%.4f | Genie-LMMSE err=%.4f\n', snr_dB, err_vamp(i_snr), err_lmmse(i_snr));
end

figure;
semilogy(snr_dB_list, err_vamp, '-x', 'LineWidth', 2); hold on;
semilogy(snr_dB_list, err_lmmse, '-o', 'LineWidth', 2);
grid on; xlabel('SNR(dB)'); ylabel('Top-1 error rate');
legend('GLM-VAMP (Exp prior+Exp noise)', 'Genie-LMMSE (known CB)', 'Location','best');
title('Known setB (8 wide beams): GLM-VAMP vs Genie-LMMSE');

%% =====================================================================
%% =========================== Local functions ==========================
%% =====================================================================

function CDL_x = load_CDL_local(csv_file, n_cluster)
    data = readmatrix(csv_file);
    data = data(~any(isnan(data),2),:);

    % Assume columns: [Cluster#, Delay, Power(dB), AoD, AoA, ZoD, ZoA]
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
    % No-toolbox exponential RNG: Exp(mean=mu):  -mu*log(U)
    [n_beams, n_cluster] = size(A_all);

    Ux = rand(n_cluster, n_samp);
    X = -repmat(mu_x,1,n_samp) .* log(max(Ux, realmin));  % Exp(mean=mu_x)

    Z = A_all * X;  % n_beams x n_samp

    snr_lin = 10^(snr_dB/10);
    P_sig = mean(Z.^2, 1);
    P_sig(P_sig < 1e-12) = 1e-12;
    P_noise = P_sig / snr_lin;

    psi_vec = sqrt(2 ./ max(P_noise, 1e-12));  % 1 x n_samp
    Uw = rand(n_beams, n_samp);
    W = -(1 ./ repmat(psi_vec, n_beams, 1)) .* log(max(Uw, realmin)); % Exp(mean=1/psi)

    Y = (Z + W).';
    psi_vec = psi_vec(:);
end

%% ===================== GLM-VAMP (x-only output) =====================

function x_hat = glmvamp_exp_exp_xonly(y, psi, lambda, A, G, prm)
    % GLM-VAMP (VAMP for GLM) style iteration [web:70][web:30]
    % Uses nullspace-preserving linear module via (gamma2/tau2 I + A A^T)^-1 (good for M<N).

    [M,N] = size(A);
    I_M = eye(M);

    max_iter = prm.max_iter;
    tol = prm.tol;
    damp = prm.damp;

    alpha_min = prm.alpha_min; alpha_max = prm.alpha_max;
    beta_min  = prm.beta_min;  beta_max  = prm.beta_max;
    gam_min = prm.gam_min; gam_max = prm.gam_max;
    tau_min = prm.tau_min; tau_max = prm.tau_max;

    % init
    r1 = 1 ./ lambda;
    gamma1 = min(max(1/mean(1./(lambda.^2)), gam_min), gam_max);

    p1 = y;                       % z <= y, start near y
    tau1 = 1;
    tau1 = min(max(tau1, tau_min), tau_max);

    for k = 1:max_iter
        r1_old = r1;
        p1_old = p1;

        % ----- x denoiser: Exp prior + Gaussian pseudo -----
        [x1, vx1] = post_x_expprior(r1, 1/gamma1, lambda);
        alpha1 = mean(gamma1 * vx1);
        alpha1 = min(max(alpha1, alpha_min), alpha_max);

        r2 = (x1 - alpha1*r1) / (1 - alpha1);
        gamma2 = gamma1*(1 - alpha1)/alpha1;
        gamma2 = min(max(gamma2, gam_min), gam_max);

        % ----- z denoiser: Exp noise channel + Gaussian pseudo -----
        [z1, vz1] = post_z_expnoise(y, p1, 1/tau1, psi);
        beta1 = mean(tau1 * vz1);
        beta1 = min(max(beta1, beta_min), beta_max);

        p2 = (z1 - beta1*p1) / (1 - beta1);
        tau2 = tau1*(1 - beta1)/beta1;
        tau2 = min(max(tau2, tau_min), tau_max);

        % ----- Linear module (nullspace-preserving) -----
        C = ((gamma2/tau2)*I_M + G) \ I_M;  % MxM
        Ar2 = A * r2;
        x2 = r2 + A.' * (C * (p2 - Ar2));
        z2 = A * x2;

        % Divergences via trace (stable for M<<N)
        T = trace(G * ((1/tau2) * C));
        alpha2 = 1 - (tau2/N)*T;
        beta2  = (tau2/M)*T;
        alpha2 = min(max(alpha2, alpha_min), alpha_max);
        beta2  = min(max(beta2,  beta_min),  beta_max);

        % ----- Extrinsic updates -----
        r1_new = (x2 - alpha2*r2) / (1 - alpha2);
        gamma1_new = gamma2*(1 - alpha2)/alpha2;

        p1_new = (z2 - beta2*p2) / (1 - beta2);
        tau1_new = tau2*(1 - beta2)/beta2;

        gamma1_new = min(max(gamma1_new, gam_min), gam_max);
        tau1_new   = min(max(tau1_new,   tau_min), tau_max);

        % damping (recommended in VAMP paper for difficult A)
        r1 = damp*r1_new + (1-damp)*r1;
        p1 = damp*p1_new + (1-damp)*p1;
        gamma1 = damp*gamma1_new + (1-damp)*gamma1;
        tau1   = damp*tau1_new   + (1-damp)*tau1;

        if (norm(r1-r1_old)/(norm(r1_old)+1e-9) < tol) && (norm(p1-p1_old)/(norm(p1_old)+1e-9) < tol)
            break;
        end
    end

    [x_hat, ~] = post_x_expprior(r1, 1/gamma1, lambda);
    x_hat = max(real(x_hat), 0);
end

function [x_post, v_post] = post_x_expprior(r, v_r, lam)
    % Exp prior, x>=0, Gaussian pseudo-measurement r = x + N(0,v_r)
    v_r = max(real(v_r), 1e-12);
    r = real(r);

    mu_shifted = r - lam .* v_r;
    sigma = sqrt(v_r);
    alpha = -mu_shifted ./ sigma;

    h = zeros(size(alpha));
    limit = 5;
    mask = alpha < limit;
    if any(mask)
        vals = alpha(mask);
        h(mask) = normpdf_local(vals) ./ max(normcdf_local(-vals), 1e-12);
    end
    if any(~mask)
        vals = alpha(~mask);
        h(~mask) = vals; % asymptotic
    end

    x_post = mu_shifted + sigma .* h;
    v_post = v_r .* (1 - h .* (alpha + h));

    x_post = max(x_post, 0);
    v_post = max(v_post, 1e-12);
end

function [z_post, vz_post] = post_z_expnoise(y, p_hat, vp, psi)
    % Exp noise: y=z+w, w~Exp(rate=psi), w>=0 => z<=y
    vp = max(real(vp), 1e-12);
    p_hat = real(p_hat);
    y = real(y);

    mu_new = p_hat + psi .* vp;
    sigma = sqrt(vp);
    alpha = (y - mu_new) ./ sigma; % upper truncation point

    rho = zeros(size(alpha));
    limit = -30;
    mask = alpha > limit;
    if any(mask)
        vals = alpha(mask);
        rho(mask) = normpdf_local(vals) ./ max(normcdf_local(vals), 1e-12);
    end
    if any(~mask)
        vals = alpha(~mask);
        rho(~mask) = -vals; % asymptotic
    end

    z_post  = mu_new - sigma .* rho;
    vz_post = vp .* (1 - rho .* (alpha + rho));
    vz_post = max(vz_post, 1e-12);
end

function y = normpdf_local(x)
    y = exp(-0.5*x.^2) / sqrt(2*pi);
end

function p = normcdf_local(x)
    % N(0,1) CDF via erfc, no toolbox needed
    p = 0.5 * erfc(-x./sqrt(2));
end
