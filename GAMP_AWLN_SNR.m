%% Simulation: Mismatch Rate vs. SNR (Exp Signal + Exp Noise + NO Norm)
% Goal: Compare GAMP strategies against standard LMMSE across SNRs
%       **Observation Model: y_{k2} = G * y_{k1} + w (K x M transformation)**
%       **NO Column Normalization applied to G**
%       **Includes Variance Clipping & Strong Damping**
%       **G is FIXED across all SNR points and trials (random once OR custom)**
% Output: A plot showing Mismatch Rate vs. SNR

clear;
clc;
close all;

%% 1. Simulation Parameters
num_trials = 1000;      % Trials per point
M = 50;                 % Signal dimension (y_{k1} dimension)
K = 25;                 % Measurement dimension (y_{k2} dimension)
sparsity_density = 0.5; % Fixed density for G

% SNR Range to sweep
SNR_vec = 0:5:35;       % Sweep from 0dB to 30dB
num_snr_points = length(SNR_vec);

% GAMP Settings
max_iter = 500;
tol = 1e-5;
num_samples_mc = 5000;  % MC samples for Method 2

% --- SURVIVAL DAMPING PARAMETERS ---
step_init = 0.05;    % Start tiny
step_min  = 0.005;
step_max  = 0.5;
step_dec  = 0.5;
step_inc  = 1.05;

% Arrays to store results
res_mean = zeros(1, num_snr_points);
res_prob = zeros(1, num_snr_points);
res_map  = zeros(1, num_snr_points);
res_lmmse= zeros(1, num_snr_points);

fprintf('Starting Simulation...\n');
fprintf('Signal Dimension (M): %d\n', M);
fprintf('Measurement Dimension (K): %d\n', K);
fprintf('G Sparsity Density: %.2f\n', sparsity_density);
fprintf('SNR Points: %s dB\n', mat2str(SNR_vec));
fprintf('Noise Type: Additive Exponential Noise\n');
fprintf('----------------------------------------\n');

total_timer = tic;

%% 1.5 FIXED transformation matrix G (choose ONE option)

% -------------------- Option 1 (DEFAULT): Fixed random G (generate once) --------------------
rng(1,"twister");                        % fixed seed for repeatability
G = sprandn(K, M, sparsity_density);     % generate once (K x M matrix)
G = abs(G);

% Column Normalization (if needed)
% col_norms = sqrt(sum(G.^2, 1));
% G = bsxfun(@rdivide, G, col_norms);

% -------------------- Option 2: Custom G (uncomment to use) --------------------
% G_custom = zeros(K,M);                 % <-- put your own K x M matrix here
% G = G_custom;

assert(all(size(G) == [K, M]), sprintf('G must be a %d x %d matrix.', K, M));

% Precompute reused terms (do once)
G2 = G.^2;
G_full = full(G);

% save('matrix_G.mat', 'G');              % Optional: save G for reuse

%% 2. Main Simulation Loops
for s_idx = 1:num_snr_points

    current_SNR = SNR_vec(s_idx);
    fprintf('Running SNR = %d dB... ', current_SNR);

    % Reset counters
    errors_mean = 0; errors_prob = 0; errors_map = 0; errors_lmmse = 0;

    snr_timer = tic;

    for trial = 1:num_trials

        % --- A. Data Generation ---
        % NOTE: G is FIXED now (no regeneration here)

        % 1. Signal y_{k1} (Exponential)
        lambda = 1 + 9 * rand(M, 1);
        mu_y = 1 ./ lambda;              % prior mean
        var_y = 1 ./ lambda.^2;          % prior variance
        y_k1_true = exprnd(mu_y);        % draw from exponential prior

        % 2. Observation y_{k2}
        z_true = G * y_k1_true;          % linear transformation
        P_sig = mean(z_true.^2);
        if P_sig < 1e-10
            P_sig = 1e-10;
        end

        % Exponential Noise: w ~ Exp(psi)
        P_noise_target = P_sig * 10^(-current_SNR/10);
        psi = sqrt(2 / P_noise_target);

        w = exprnd(1/psi, K, 1);
        y_k2 = z_true + w;

        [~, true_max_idx] = max(y_k1_true);

        % --- B. Run Sum-Product GAMP ---
        y_hat = mu_y;
        vy = var_y;
        s_hat = zeros(K, 1);

        step = step_init;
        diff_norm_prev = 1e10;

        % Store final states
        r_final = zeros(M, 1);
        vr_final = zeros(M, 1);

        for t = 1:max_iter
            % 1. Output Linear
            vp = G2 * vy;

            % Variance Clipping (Upper Bound)
            vp = min(max(real(vp), 1e-10), 1e5);

            p_hat = G * y_hat - vp .* s_hat;

            % 2. Output Nonlinear (Exponential Noise)
            [z_post, vz_post] = estim_output_exp_noise(y_k2, p_hat, vp, psi);
            s_hat_new = (z_post - p_hat) ./ vp;
            vs_new    = (1 - vz_post ./ vp) ./ vp;

            % 3. Input Linear
            vr = 1 ./ (G2' * vs_new);

            % Variance Clipping (Upper Bound)
            vr = min(max(real(vr), 1e-10), 1e5);

            r_hat = y_hat + vr .* (G' * s_hat_new);

            % 4. Input Nonlinear (Exponential Signal Prior)
            [y_hat_new, vy_new] = estim_input_exponential(r_hat, vr, lambda);

            % --- Robust Damping Logic ---
            diff_norm = norm(y_hat_new - y_hat);

            if t > 1 && (isnan(diff_norm) || diff_norm > 1e8)
                step = step_min;
                y_hat = mu_y;
                vy = var_y;
                s_hat = zeros(K,1);
            else
                s_hat = step * s_hat_new + (1-step) * s_hat;
                y_hat = step * y_hat_new + (1-step) * y_hat;
                vy    = step * vy_new    + (1-step) * vy;
                vy    = max(real(vy), 1e-10);

                if t > 1
                    if diff_norm < diff_norm_prev
                        step = min(step * step_inc, step_max);
                    else
                        step = max(step * step_dec, step_min);
                    end
                end
                diff_norm_prev = diff_norm;
            end

            r_final = r_hat;
            vr_final = vr;

            if diff_norm / norm(y_hat + 1e-9) < tol
                break;
            end
        end

        % --- Method 1: GAMP Posterior Mean (MMSE) ---
        [~, idx_gamp_mean] = max(y_hat);

        % --- Method 2: GAMP Posterior Probability (Ranking with MC) ---
        mu_eff = r_final - lambda .* vr_final;
        sigma_eff = sqrt(vr_final);
        alpha = -mu_eff ./ sigma_eff;
        Phi_alpha = normcdf(alpha);
        Z_valid = max(1 - Phi_alpha, 1e-10);
        U = rand(M, num_samples_mc);
        P_target = min(max(bsxfun(@plus, Phi_alpha, bsxfun(@times, U, Z_valid)), 1e-10), 1-1e-10);
        Y_samples = mu_eff + sigma_eff .* norminv(P_target);
        [~, max_indices] = max(Y_samples, [], 1);
        idx_gamp_prob = mode(max_indices);

        % --- Method 3: GAMP Posterior Mode (MAP) ---
        y_map = max(0, r_final - lambda .* vr_final);
        [~, idx_gamp_map] = max(y_map);

        % --- Method 4: Linear MMSE (Baseline) ---
        Cy = diag(var_y);
        noise_var = 1/psi^2;
        noise_mean = 1/psi;
        Cw = noise_var * eye(K);
        mean_y_k2 = G_full * mu_y + noise_mean;
        Cyy = G_full * Cy * G_full' + Cw;
        Cxy = Cy * G_full';
        y_lmmse = mu_y + Cxy * ((Cyy + 1e-10*eye(K)) \ (y_k2 - mean_y_k2));
        [~, idx_lmmse] = max(y_lmmse);

        % --- Check Mismatch ---
        if idx_gamp_mean ~= true_max_idx, errors_mean = errors_mean + 1; end
        if idx_gamp_prob ~= true_max_idx, errors_prob = errors_prob + 1; end
        if idx_gamp_map  ~= true_max_idx, errors_map  = errors_map  + 1; end
        if idx_lmmse     ~= true_max_idx, errors_lmmse= errors_lmmse+ 1; end
    end

    % Store Results
    res_mean(s_idx) = errors_mean / num_trials;
    res_prob(s_idx) = errors_prob / num_trials;
    res_map(s_idx)  = errors_map  / num_trials;
    res_lmmse(s_idx)= errors_lmmse/ num_trials;

    fprintf('Done (%.2f s)\n', toc(snr_timer));
end

toc(total_timer);

%% 3. Plotting Results
figure('Name', 'Mismatch Rate vs SNR', 'Color', 'w');
hold on;
grid on;
box on;

plot(SNR_vec, res_mean,  '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'GAMP Mean');
plot(SNR_vec, res_prob,  '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'GAMP Prob (Ranking)');
% plot(SNR_vec, res_map,   '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'GAMP MAP');
plot(SNR_vec, res_lmmse, '--x','LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'LMMSE (Baseline)');

xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Mismatch Probability', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Mismatch Rate vs. SNR (M =%d, K =%d, Density =%.1f, Exp Noise)', M, K, sparsity_density), 'FontSize', 14);
legend('show', 'Location', 'best', 'FontSize', 10);
ylim([0, 1]);

%% Display Transformation Matrix G
fprintf('\n========== TRANSFORMATION MATRIX G ==========\n');
fprintf('Matrix G (size: %d x %d, sparsity: %.2f%%):\n', K, M, 100 * nnz(G) / numel(G));
if nnz(G) / numel(G) < 0.3  % If sparse, show sparse format
    fprintf('Showing in sparse format (row, col, value):\n');
    [row, col, val] = find(G);
    for k = 1:min(20, length(row))  % Show first 20 non-zero entries
        fprintf('  (%2d, %2d) = %.4f\n', row(k), col(k), val(k));
    end
    if length(row) > 20
        fprintf('  ... (%d more non-zero entries)\n', length(row) - 20);
    end
else  % If dense, show full matrix
    disp(full(G));
end
fprintf('=============================================\n');

%% --- Helper Functions ---
function [y_post, v_post] = estim_input_exponential(r, v_r, lam)
    % Input estimation: y_{k1,j} ~ lower-truncated Gaussian
    % Posterior of y_{k1,j} given measurement r_j and variance v_r
    
    v_r = max(real(v_r), 1e-10);
    r = real(r);

    mu_shifted = r - lam .* v_r;
    sigma = sqrt(v_r);
    alpha = -mu_shifted ./ sigma;

    h = zeros(size(alpha));
    limit = 5;
    mask_normal = alpha < limit;
    mask_tail = ~mask_normal;

    if any(mask_normal)
        vals = alpha(mask_normal);
        h(mask_normal) = normpdf(vals) ./ normcdf(-vals);
    end
    if any(mask_tail)
        vals = alpha(mask_tail);
        h(mask_tail) = vals;
    end

    y_post = mu_shifted + sigma .* h;
    v_post = v_r .* (1 - h .* (alpha + h));
    y_post = max(y_post, 0);
    v_post = max(v_post, 1e-10);
end

function [z_post, vz_post] = estim_output_exp_noise(y_k2, p_hat, vp, psi)
    % Output estimation: z_i ~ upper-truncated Gaussian (due to exp noise)
    % Posterior of z_i = (G*y_{k1})_i given observation y_{k2,i} and variance vp
    
    vp = max(real(vp), 1e-10);
    p_hat = real(p_hat);

    mu_new = p_hat + psi .* vp;
    sigma = sqrt(vp);
    alpha = (y_k2 - mu_new) ./ sigma;

    rho = zeros(size(alpha));
    limit = -30;
    mask_normal = alpha > limit;
    mask_tail = ~mask_normal;

    if any(mask_normal)
        vals = alpha(mask_normal);
        rho(mask_normal) = normpdf(vals) ./ normcdf(vals);
    end
    if any(mask_tail)
        vals = alpha(mask_tail);
        rho(mask_tail) = -vals;
    end

    z_post = mu_new - sigma .* rho;
    vz_post = vp .* (1 - rho .* (alpha + rho));
    vz_post = max(vz_post, 1e-10);
end
