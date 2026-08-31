%% run_setB_MADGAMP_Exp_vs_LMMSE_FULL.m
clear; clc; close all;
rng(1,'twister');

%% ===================== Settings =====================
csv_file = 'CDL_A.csv';
n_cluster = 21;

snr_dB_training = 25;
snr_dB_list = -10:5:25;

n_train = 5000;
n_test  = 5000;

n_beams_w = 8;
n_beams_n = 32;
n_beams   = n_beams_w + n_beams_n;

k2 = n_beams_w;
k1 = n_beams - k2;
obs_indices = (k1+1):n_beams;

use_column_normalization = true;

% ----- MAD-GAMP params -----
prm.max_iter = 800;
prm.tol = 1e-6;
prm.beta     = 0.5;
prm.beta_max = 1.0;
prm.beta_min = 0.01;
prm.Gpass    = 1.10;
prm.Gfail    = 0.50;
prm.Tbeta    = 0;
prm.cost_eps = 1e-12;
prm.tau_min = 1e-12;
prm.tau_max = 1e8;

mr.gamma = 1.0;

%% ===================== Load CDL =====================
assert(exist(csv_file,'file')==2, 'CSV not found: %s', csv_file);
CDL_x = load_CDL_local(csv_file, n_cluster);
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

%% ===================== Build A =====================
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
mu_x = 2*(sigma.^2);
lambda = 1 ./ max(mu_x, 1e-12);

%% ===================== Column normalization =====================
if use_column_normalization
    col_norms = sqrt(sum(A_obs.^2,1)).';
    col_norms(col_norms < 1e-12) = 1;
    A_obs_use = A_obs ./ (col_norms.');
    lambda_use = lambda ./ col_norms;
else
    A_obs_use = A_obs;
    lambda_use = lambda;
end

%% ===================== Mean-removal augmentation =====================
[Atil, mrinfo] = mean_removal_augment(A_obs_use, mr.gamma);

%% ===================== Train LMMSE: x from yB =====================
[Xtr, Ytr_exp, ~] = generate_xy_vectorized_expnoise(A_all, mu_x, snr_dB_training, n_train);
YBtr = Ytr_exp(:, obs_indices);

mu_x_tr  = mean(Xtr, 2);
mu_yB_tr = mean(YBtr, 1).';

Xc  = Xtr - mu_x_tr;
YBc = (YBtr.' - mu_yB_tr);

Sigma_xyB  = (Xc * YBc.') / (n_train-1);
Sigma_yByB = (YBc * YBc.') / (n_train-1);
W_lmmse = Sigma_xyB / (Sigma_yByB + 1e-10*eye(k2));

%% ===================== Evaluate =====================
L = length(snr_dB_list);

errx_lmmse = zeros(L,1);
errx_mad   = zeros(L,1);
nmsx_lmmse = zeros(L,1);
nmsx_mad   = zeros(L,1);

errz_lmmse = zeros(L,1);
errz_mad   = zeros(L,1);
nmsz_lmmse = zeros(L,1);
nmsz_mad   = zeros(L,1);

for i_snr = 1:L
    snr_dB = snr_dB_list(i_snr);

    [Xte, Yte_exp, psi_vec] = generate_xy_vectorized_expnoise(A_all, mu_x, snr_dB, n_test);
    YB_exp = Yte_exp(:, obs_indices);

    cntx_l=0; cntx_g=0;
    cntz_l=0; cntz_g=0;
    sex_l=0;  sex_g=0;  sx=0;
    sez_l=0;  sez_g=0;  sz=0;

    for t = 1:n_test
        x_true = Xte(:,t);
        z_true = A_obs_use * x_true;

        [~, jx_true] = max(x_true);
        [~, jz_true] = max(z_true);

        yB = YB_exp(t,:).';
        psi = psi_vec(t);

        % LMMSE
        x_hat_l = mu_x_tr + W_lmmse*(yB - mu_yB_tr);
        z_hat_l = A_obs_use * x_hat_l;

        % MAD-GAMP (mean-removal)
        x_hat_g = madgamp_expnoise_stable(yB, psi, Atil, mrinfo, lambda_use, prm);
        z_hat_g = A_obs_use * x_hat_g;

        if (snr_dB==25) && (t==1)
            fprintf('--- debug mean-removal @25dB sample1 ---\n');
            fprintf('mean(A_obs_use)=%.3e, mean(Atil)=%.3e\n', mean(A_obs_use(:)), mean(Atil(:)));
            fprintf('MADGAMP: mean(y)=%.3e, mean(Ax)=%.3e, mean(y-Ax)=%.3e\n', mean(yB), mean(A_obs_use*x_hat_g), mean(yB - A_obs_use*x_hat_g));
        end

        % argmax(x)
        [~, jx_l] = max(x_hat_l);
        [~, jx_g] = max(x_hat_g);
        cntx_l = cntx_l + (jx_l ~= jx_true);
        cntx_g = cntx_g + (jx_g ~= jx_true);

        % argmax(z)
        [~, jz_l] = max(z_hat_l);
        [~, jz_g] = max(z_hat_g);
        cntz_l = cntz_l + (jz_l ~= jz_true);
        cntz_g = cntz_g + (jz_g ~= jz_true);

        % NMSE(x)
        sex_l = sex_l + norm(x_true - x_hat_l)^2;
        sex_g = sex_g + norm(x_true - x_hat_g)^2;
        sx    = sx    + norm(x_true)^2;

        % NMSE(z)
        sez_l = sez_l + norm(z_true - z_hat_l)^2;
        sez_g = sez_g + norm(z_true - z_hat_g)^2;
        sz    = sz    + norm(z_true)^2;
    end

    errx_lmmse(i_snr) = cntx_l/n_test;
    errx_mad(i_snr)   = cntx_g/n_test;
    nmsx_lmmse(i_snr) = sex_l/max(sx,1e-12);
    nmsx_mad(i_snr)   = sex_g/max(sx,1e-12);

    errz_lmmse(i_snr) = cntz_l/n_test;
    errz_mad(i_snr)   = cntz_g/n_test;
    nmsz_lmmse(i_snr) = sez_l/max(sz,1e-12);
    nmsz_mad(i_snr)   = sez_g/max(sz,1e-12);

    fprintf(['SNR=%5.1f dB | argmax(x): MAD=%.4f LMMSE=%.4f | xNMSE: MAD=%.3e LMMSE=%.3e | ' ...
             'argmax(z): MAD=%.4f LMMSE=%.4f | zNMSE: MAD=%.3e LMMSE=%.3e\n'], ...
        snr_dB, errx_mad(i_snr), errx_lmmse(i_snr), nmsx_mad(i_snr), nmsx_lmmse(i_snr), ...
        errz_mad(i_snr), errz_lmmse(i_snr), nmsz_mad(i_snr), nmsz_lmmse(i_snr));
end

%% ===================== Plots =====================
figure; semilogy(snr_dB_list, errx_mad,'-x', snr_dB_list, errx_lmmse,'-o','LineWidth',2);
grid on; xlabel('SNR(dB)'); ylabel('Top-1 error'); legend('MAD-GAMP','LMMSE','Location','best'); title('argmax(x)');

figure; semilogy(snr_dB_list, nmsx_mad,'-x', snr_dB_list, nmsx_lmmse,'-o','LineWidth',2);
grid on; xlabel('SNR(dB)'); ylabel('x-NMSE'); legend('MAD-GAMP','LMMSE','Location','best'); title('x-NMSE');

figure; semilogy(snr_dB_list, errz_mad,'-x', snr_dB_list, errz_lmmse,'-o','LineWidth',2);
grid on; xlabel('SNR(dB)'); ylabel('Top-1 error'); legend('MAD-GAMP','LMMSE','Location','best'); title('argmax(z)');

figure; semilogy(snr_dB_list, nmsz_mad,'-x', snr_dB_list, nmsz_lmmse,'-o','LineWidth',2);
grid on; xlabel('SNR(dB)'); ylabel('z-NMSE'); legend('MAD-GAMP','LMMSE','Location','best'); title('z-NMSE');

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
    M = params(1); N = params(2);
    AoPeak = params(3); ZoPeak = params(4);
    sep = 1/2;
    ppM_deg = sep * 360 * sind(AoPeak);
    ppN_deg = sep * 360 * cosd(ZoPeak) / sqrt(1 - (sind(ZoPeak))^2 * (sind(AoPeak))^2);
    phi_3dB=90; theta_3dB=90; A_m=30; SLA_v=30; G_E_max=5.5;
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
    ppM = ppM_deg*pi/180; ppN = ppN_deg*pi/180;
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

function [X, Y, psi_vec] = generate_xy_vectorized_expnoise(A_all, mu_x, snr_dB, n_samp)
    [n_beams, n_cluster] = size(A_all);
    Ux = rand(n_cluster, n_samp);
    X = -repmat(mu_x,1,n_samp) .* log(max(Ux, realmin));
    Z = A_all * X;

    snr_lin = 10^(snr_dB/10);
    P_sig = mean(Z.^2, 1);
    P_sig(P_sig < 1e-12) = 1e-12;
    P_noise = P_sig / snr_lin;

    psi_vec = sqrt(2 ./ max(P_noise, 1e-12));
    Uw = rand(n_beams, n_samp);
    W = -(1 ./ repmat(psi_vec, n_beams, 1)) .* log(max(Uw, realmin));
    Y = (Z + W).';
    psi_vec = psi_vec(:);
end

function [Atil, info] = mean_removal_augment(A, gamma)
% Mean-removal augmentation idea from MAD-GAMP literature [web:200]
    [M,N] = size(A);
    mu = mean(A(:));
    rowMean = mean(A,2);
    colMean = mean(A,1).';
    Abar = A - rowMean*ones(1,N) - ones(M,1)*colMean.' + mu;

    b12 = gamma * sqrt(mean(Abar(:).^2) + 1e-12);
    b13 = b12; b21 = b12; b31 = b12;

    Atil = [ Abar,          b12*ones(M,1),  b13*rowMean;
             b21*ones(1,N), 0,              0;
             b31*colMean.', 0,              0 ];

    info.M=M; info.N=N;
end

function x_hat = madgamp_expnoise_stable(y, psi, Atil, info, lambda, prm)
% AD-GAMP on augmented system (first M outputs use Exp noise; last 2 dummy)
% Uses stable inverse Mills ratio via erfcx [web:334]
    M=info.M; N=info.N;
    Mt=M+2; Nt=N+2;

    tau_min=prm.tau_min; tau_max=prm.tau_max;
    Tmax=prm.max_iter; tol=prm.tol;
    beta=prm.beta; beta_max=prm.beta_max; beta_min=prm.beta_min;
    Gpass=prm.Gpass; Gfail=prm.Gfail; Tbeta=prm.Tbeta; cost_eps=prm.cost_eps;

    bigVar = 1e12;

    x_hat = [1./lambda; 0; 0];
    x_var = [1./(lambda.^2); bigVar; bigVar];
    s_hat = zeros(Mt,1);
    A2 = Atil.^2;

    J_hist = inf(Tmax+1,1); J_hist(1)=inf;
    sigw2_dummy = 1e12;

    t=1;
    while t<=Tmax
        x_old=x_hat; v_old=x_var; s_old=s_hat;

        tau_p = A2*x_var;
        tau_p = min(max(tau_p, tau_min), tau_max);
        p_hat = Atil*x_hat - tau_p.*s_hat;

        z_hat=zeros(Mt,1); z_var=zeros(Mt,1);
        [z_hat(1:M), z_var(1:M)] = post_z_expnoise_stable(y, p_hat(1:M), tau_p(1:M), psi);

        z_hat(M+1:Mt) = (sigw2_dummy.*p_hat(M+1:Mt)) ./ (sigw2_dummy + tau_p(M+1:Mt));
        z_var(M+1:Mt) = (sigw2_dummy.*tau_p(M+1:Mt)) ./ (sigw2_dummy + tau_p(M+1:Mt));

        z_var = min(max(z_var, tau_min), tau_p);
        s_new = (z_hat - p_hat)./tau_p;

        tau_s = (1./tau_p).*max(1 - z_var./tau_p, 0);
        tau_s = min(max(tau_s, tau_min), tau_max);

        denom = (A2.'*tau_s);
        denom = max(denom, tau_min);
        tau_r = 1./denom;
        tau_r = min(max(tau_r, tau_min), tau_max);

        r_hat = x_hat + tau_r.*(Atil.'*s_new);

        [x_new1, v_new1] = post_x_expprior(r_hat(1:N), tau_r(1:N), lambda);
        x_new = [x_new1; r_hat(N+1); r_hat(N+2)];
        v_new = [v_new1; tau_r(N+1); tau_r(N+2)];
        v_new = min(max(v_new, tau_min), tau_max);

        x_cand = beta*x_new + (1-beta)*x_hat;
        v_cand = beta*v_new + (1-beta)*x_var;
        s_cand = beta*s_new + (1-beta)*s_hat;

        if any(~isfinite(x_cand)) || any(~isfinite(v_cand)) || any(~isfinite(s_cand))
            pass=false;
        else
            z_cand = Atil*x_cand;
            z1 = z_cand(1:M);
            viol = max(z1 - y, 0);
            J = sum(lambda.*max(x_cand(1:N),0)) + psi*sum(max(y - z1,0)) + 1e6*sum(viol);

            J_hist(t+1)=J;
            t0=max(1,t-Tbeta);
            worst_recent=max(J_hist(t0:t));
            pass = (J <= worst_recent*(1+cost_eps)) || (beta <= beta_min + 1e-15);
        end

        if pass
            x_hat = x_cand;
            x_hat(1:N) = max(real(x_hat(1:N)),0);
            x_var = v_cand;
            s_hat = s_cand;

            if norm(x_hat-x_old)/(norm(x_old)+1e-9) < tol
                break;
            end
            beta = min(beta_max, beta*Gpass);
            t=t+1;
        else
            x_hat=x_old; x_var=v_old; s_hat=s_old;
            beta = max(beta_min, beta*Gfail);
        end
    end

    x_hat = max(real(x_hat(1:N)),0);
end

function [x_post, v_post] = post_x_expprior(r, v_r, lam)
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
        h(~mask) = vals;
    end

    x_post = mu_shifted + sigma .* h;
    v_post = v_r .* (1 - h .* (alpha + h));
    x_post = max(x_post, 0);
    v_post = max(v_post, 1e-12);
end

function [z_post, vz_post] = post_z_expnoise_stable(y, p_hat, vp, psi)
    vp = max(real(vp), 1e-12);
    p_hat = real(p_hat);
    y = real(y);

    mu = p_hat + psi .* vp;
    sigma = sqrt(vp);
    alpha = (y - mu) ./ sigma;

    t = -alpha ./ sqrt(2);
    erfcx_t = erfcx(t);
    erfcx_t = max(erfcx_t, 1e-300);
    lam = sqrt(2/pi) ./ erfcx_t;  % stable phi/Phi [web:334]

    z_post  = mu - sigma .* lam;
    vz_post = vp .* (1 - lam .* (alpha + lam));
    vz_post = max(vz_post, 1e-12);
end

function y = normpdf_local(x)
    y = exp(-0.5*x.^2) / sqrt(2*pi);
end

function p = normcdf_local(x)
    p = 0.5 * erfc(-x./sqrt(2));
end
