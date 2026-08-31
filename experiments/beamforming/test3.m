%% run_setB_ADGAMP_vs_LMMSE_argmax_nmse_singlefile.m
% Self-contained: only needs CDL_A.csv
%
% Compare estimating latent exponentials x (n_cluster=21) from y_B (8 wide beams).
% Metrics:
%   (1) argmax(x) top-1 error rate
%   (2) x-NMSE
%
% Baseline: LMMSE learned from training pairs (x, yB)
% Proposed: AD-GAMP (adaptive damping) for Exp prior + Exp noise output channel
%
% === Fixes in this version ===
% (A) post_x_expprior: FIX variance formula for lower-truncated normal (x>=0)
% (B) post_z_expnoise: use erfcx-stable computation of rho=phi/Phi for very negative alpha
% (C) post_x_expprior: use erfcx-stable computation of h=phi/Phi(-alpha)
% (D) column normalization: unnormalize x_hat before argmax/NMSE

clear; clc; close all;
rng(1,'twister');

%% ===================== User settings =====================
csv_file = 'CDL_A.csv';
n_cluster = 21;
snr_dB_training = 25;
snr_dB_list = -10:5:25;
n_train = 5000;     % for LMMSE on x|yB
n_test  = 5000;     % per SNR
n_beams_w = 8;
n_beams_n = 32;
n_beams   = n_beams_w + n_beams_n;
k2 = n_beams_w;
k1 = n_beams - k2;
obs_indices = (k1+1):n_beams;  % last 8 beams are wide
use_column_normalization = true;

% ----- AD-GAMP params -----
gamp.max_iter = 500;
gamp.tol = 1e-6;

% adaptive damping beta(t)
gamp.beta     = 0.9;
gamp.beta_max = 1.0;
gamp.beta_min = 0.10;
gamp.Gpass    = 1.05;   % accepted -> beta*=Gpass
gamp.Gfail    = 0.80;   % rejected -> beta*=Gfail, retry same iter
gamp.Tbeta    = 5;      % compare to recent window
gamp.cost_eps = 1e-6;

% numeric guards
gamp.tau_min = 1e-12;
gamp.tau_max = 1e8;

%% ===================== Load CDL =====================
assert(exist(csv_file,'file')==2, 'CSV not found: %s', csv_file);
CDL_x = load_CDL_local(csv_file, n_cluster);  % [power_linear, AoD, AoA, ZoD, ZoA]
AoD = CDL_x(:,2);  ZoD = CDL_x(:,4);
AoA = CDL_x(:,3);  ZoA = CDL_x(:,5);

%% ===================== Beam setup =====================
range_aop = 60; range_zop = 30; range_aop_fine = 30;
aop_tx = 0; zop_tx = 90;
tx_params_n = generate_tx_para_local(8,4, 8,4, range_aop_fine, range_zop, aop_tx, zop_tx);
tx_params_w = generate_tx_para_local(4,2, 4,2, range_aop,      range_zop, aop_tx, zop_tx);
tx_params = [tx_params_n; tx_params_w];

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
A_obs = A_all(obs_indices,:);  % 8x21

%% ===================== Prior for latent x =====================
sigma = CDL_x(:,1);
mu_x = 2*(sigma.^2);                  % Exp mean
lambda = 1 ./ max(mu_x, 1e-12);       % Exp rate

%% ===================== Column normalization =====================
if use_column_normalization
    col_norms = sqrt(sum(A_obs.^2,1)).';
    col_norms(col_norms < 1e-12) = 1;
    A_obs_use = A_obs ./ (col_norms.');
    lambda_use = lambda ./ col_norms; % x' = d*x => rate' = rate/d
else
    A_obs_use = A_obs;
    lambda_use = lambda;
    col_norms  = ones(n_cluster,1);
end

%% ===================== Train LMMSE: x from yB =====================
[Xtr, Ytr, ~] = generate_xy_vectorized(A_all, mu_x, snr_dB_training, n_train);
YBtr = Ytr(:, obs_indices);

mu_x_tr  = mean(Xtr, 2);
mu_yB_tr = mean(YBtr, 1).';

Xc  = Xtr - mu_x_tr;                 % 21 x n_train
YBc = (YBtr.' - mu_yB_tr);           % 8  x n_train

Sigma_xyB  = (Xc * YBc.') / (n_train-1);      % 21x8
Sigma_yByB = (YBc * YBc.') / (n_train-1);     % 8x8
W_lmmse = Sigma_xyB / (Sigma_yByB + 1e-10*eye(k2));  % 21x8

%% ===================== Evaluate: argmax + NMSE =====================
err_lmmse = zeros(length(snr_dB_list),1);
err_gamp  = zeros(length(snr_dB_list),1);
nmse_lmmse = zeros(length(snr_dB_list),1);
nmse_gamp  = zeros(length(snr_dB_list),1);

for i_snr = 1:length(snr_dB_list)
    snr_dB = snr_dB_list(i_snr);
    [Xte, Yte, psi_vec] = generate_xy_vectorized(A_all, mu_x, snr_dB, n_test);
    YBte = Yte(:, obs_indices);

    cnt_l = 0; cnt_g = 0;
    se_l = 0;  se_g = 0;  sx = 0;

    for t = 1:n_test
        x_true = Xte(:,t);
        yB = YBte(t,:).';
        psi = psi_vec(t);

        [~, j_true] = max(x_true);

        % LMMSE
        x_hat_l = mu_x_tr + W_lmmse * (yB - mu_yB_tr);
        [~, j_l] = max(x_hat_l);

        % AD-GAMP (solves for x' if column-normalized)
        x_hat_g = gamp_expprior_expnoise_AD(yB, psi, A_obs_use, lambda_use, gamp);

        % IMPORTANT: unnormalize back to x (so metrics match x_true)
        if use_column_normalization
            x_hat_g = x_hat_g ./ col_norms;
        end

        [~, j_g] = max(x_hat_g);

        cnt_l = cnt_l + (j_l ~= j_true);
        cnt_g = cnt_g + (j_g ~= j_true);

        se_l = se_l + norm(x_true - x_hat_l)^2;
        se_g = se_g + norm(x_true - x_hat_g)^2;
        sx   = sx   + norm(x_true)^2;
    end

    err_lmmse(i_snr) = cnt_l / n_test;
    err_gamp(i_snr)  = cnt_g / n_test;
    nmse_lmmse(i_snr) = se_l / max(sx, 1e-12);
    nmse_gamp(i_snr)  = se_g / max(sx, 1e-12);

    fprintf('SNR=%5.1f dB | argmax err: GAMP=%.4f LMMSE=%.4f | x-NMSE: GAMP=%.3e LMMSE=%.3e\n', ...
        snr_dB, err_gamp(i_snr), err_lmmse(i_snr), nmse_gamp(i_snr), nmse_lmmse(i_snr));
end

%% ===================== Plot 1: argmax error =====================
figure;
semilogy(snr_dB_list, err_gamp, '-x', 'LineWidth', 2); hold on;
semilogy(snr_dB_list, err_lmmse, '-o', 'LineWidth', 2);
grid on; xlabel('SNR(dB)'); ylabel('Top-1 error rate on argmax(x)');
legend('AD-GAMP', 'LMMSE (learned x|yB)', 'Location','best');
title('Argmax(x) classification from setB');

%% ===================== Plot 2: NMSE =====================
figure;
semilogy(snr_dB_list, nmse_gamp, '-x', 'LineWidth', 2); hold on;
semilogy(snr_dB_list, nmse_lmmse, '-o', 'LineWidth', 2);
grid on; xlabel('SNR(dB)'); ylabel('x-NMSE');
legend('AD-GAMP', 'LMMSE (learned x|yB)', 'Location','best');
title('Latent x estimation NMSE from setB');

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

function [X, Y, psi_vec] = generate_xy_vectorized(A_all, mu_x, snr_dB, n_samp)
    % Inverse transform sampling for exponential: X=-m*log(U)
    [n_beams, n_cluster] = size(A_all);
    Ux = rand(n_cluster, n_samp);
    X = -repmat(mu_x,1,n_samp) .* log(max(Ux, realmin));
    Z = A_all * X;

    snr_lin = 10^(snr_dB/10);
    P_sig = mean(Z.^2, 1);
    P_sig(P_sig < 1e-12) = 1e-12;
    P_noise = P_sig / snr_lin;
    psi_vec = sqrt(2 ./ max(P_noise, 1e-12));   % your original mapping

    Uw = rand(n_beams, n_samp);
    W = -(1 ./ repmat(psi_vec, n_beams, 1)) .* log(max(Uw, realmin)); % Exp(rate=psi)
    Y = (Z + W).';
    psi_vec = psi_vec(:);
end

%% ===================== AD-GAMP core =====================
function x_hat = gamp_expprior_expnoise_AD(y, psi, A, lambda, prm)
% Adaptive damping + numeric guards for generic A
% Prior: x_j ~ Exp(rate=lambda_j), x>=0
% Likelihood: y = z + w, w~Exp(rate=psi), w>=0 => z<=y

    [M,~] = size(A);
    tau_min = prm.tau_min;
    tau_max = prm.tau_max;
    Tmax = prm.max_iter;
    tol  = prm.tol;

    beta     = prm.beta;
    beta_max = prm.beta_max;
    beta_min = prm.beta_min;
    Gpass    = prm.Gpass;
    Gfail    = prm.Gfail;
    Tbeta    = prm.Tbeta;
    cost_eps = prm.cost_eps;

    x_hat = 1 ./ lambda;
    x_var = 1 ./ (lambda.^2);
    s_hat = zeros(M,1);

    A2 = A.^2;
    J_hist = inf(Tmax+1,1);
    J_hist(1) = inf;

    t = 1;
    while t <= Tmax
        x_old = x_hat; v_old = x_var; s_old = s_hat;

        % ---- output linear ----
        tau_p = A2 * x_var;
        tau_p = min(max(tau_p, tau_min), tau_max);
        p_hat = A * x_hat - tau_p .* s_hat;

        % ---- output nonlinear (exp noise) ----
        [z_hat, z_var] = post_z_expnoise(y, p_hat, tau_p, psi);

        z_var = min(max(z_var, tau_min), tau_p);

        s_new = (z_hat - p_hat) ./ tau_p;
        tau_s = (1 ./ tau_p) .* max(1 - z_var ./ tau_p, 0);
        tau_s = min(max(tau_s, tau_min), tau_max);

        denom = (A2.' * tau_s);
        denom = max(denom, tau_min);

        tau_r = 1 ./ denom;
        tau_r = min(max(tau_r, tau_min), tau_max);

        r_hat = x_hat + tau_r .* (A.' * s_new);

        % ---- input nonlinear (exp prior) ----
        [x_new, v_new] = post_x_expprior(r_hat, tau_r, lambda);
        v_new = min(max(v_new, tau_min), tau_max);

        % damping
        x_cand = beta*x_new + (1-beta)*x_hat;
        v_cand = beta*v_new + (1-beta)*x_var;
        s_cand = beta*s_new + (1-beta)*s_hat;

        % pass/fail
        if any(~isfinite(x_cand)) || any(~isfinite(v_cand)) || any(~isfinite(s_cand))
            pass = false;
        else
            z_cand = A * x_cand;
            viol = max(z_cand - y, 0);
            J = sum(lambda .* max(x_cand,0)) + psi*sum(max(y - z_cand, 0)) + 1e6*sum(viol);
            J_hist(t+1) = J;

            t0 = max(1, t - Tbeta);
            worst_recent = max(J_hist(t0:t));
            pass = (J <= worst_recent*(1+cost_eps)) || (beta <= beta_min + 1e-15);
        end

        if pass
            x_hat = max(real(x_cand),0);
            x_var = v_cand;
            s_hat = s_cand;

            if norm(x_hat - x_old)/(norm(x_old)+1e-9) < tol
                return;
            end

            beta = min(beta_max, beta*Gpass);
            t = t + 1;
        else
            x_hat = x_old; x_var = v_old; s_hat = s_old;
            beta = max(beta_min, beta*Gfail);
        end
    end

    x_hat = max(real(x_hat),0);
end

function [x_post, v_post] = post_x_expprior(r, v_r, lam)
% Posterior of x given r ~ N(x, v_r) and prior x~Exp(lam) with x>=0
% Equivalent to a lower-truncated normal with shifted mean mu_shifted = r - lam*v_r

    v_r = max(real(v_r), 1e-12);
    r = real(r);

    mu_shifted = r - lam .* v_r;
    sigma = sqrt(v_r);

    % lower truncation at 0 => alpha = (0 - mu)/sigma
    alpha = -mu_shifted ./ sigma;

    % h(alpha) = phi(alpha)/Phi(-alpha) computed stably with erfcx
    % Phi(-alpha) = 0.5*erfc(alpha/sqrt(2))
    u = alpha ./ sqrt(2);
    h = sqrt(2/pi) ./ max(erfcx(u), 1e-300);

    x_post = mu_shifted + sigma .* h;

    % variance for lower truncation (x>=0)
    v_post = v_r .* (1 + alpha .* h - h.^2);

    x_post = max(x_post, 0);
    v_post = max(v_post, 1e-12);
end

function [z_post, vz_post] = post_z_expnoise(y, p_hat, vp, psi)
% Posterior of z given pseudo-prior z ~ N(p_hat, vp) and likelihood y=z+w, w~Exp(psi), w>=0
% => support z<=y; posterior proportional to N(z; p_hat, vp)*exp(psi*z)*1(z<=y)
% => upper-truncated normal with mean mu_new = p_hat + psi*vp

    vp = max(real(vp), 1e-12);
    p_hat = real(p_hat);
    y = real(y);

    mu_new = p_hat + psi .* vp;
    sigma = sqrt(vp);
    alpha = (y - mu_new) ./ sigma;

    % rho(alpha) = phi(alpha)/Phi(alpha) computed stably with erfcx
    % Phi(alpha) = 0.5*erfc(-alpha/sqrt(2))
    u = -alpha ./ sqrt(2);
    rho = sqrt(2/pi) ./ max(erfcx(u), 1e-300);

    z_post  = mu_new - sigma .* rho;
    vz_post = vp .* (1 - rho .* (alpha + rho));
    vz_post = max(vz_post, 1e-12);
end
