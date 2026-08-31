%% Simulation: Mismatch Probability Analysis (Exp Signal + AWGN + SNR Control)
% Goal: Compare GAMP strategies against standard LMMSE under AWGN
%       controlled by SNR.
% Method 1: GAMP Posterior Mean (MMSE)
% Method 2: GAMP Posterior Probability (Ranking)
% Method 3: GAMP Posterior Mode (MAP)
% Method 4: Linear MMSE (Baseline)

clear;
clc;
close all;

%% 1. Simulation Parameters
num_trials = 1000;  % Total trials
N = 50;             % Signal dimension
M = 25;             % Measurement dimension
sparsity_density = 0.3;

% SNR Parameter
SNR_dB = 20;        % Set Target SNR in dB

% GAMP Settings
max_iter = 500;
tol = 1e-5;
damp = 0.85;
num_samples_mc = 10000; % Monte Carlo samples for Method 2

% Counters for Mismatch
errors_mean = 0;  % Method 1
errors_prob = 0;  % Method 2
errors_map  = 0;  % Method 3
errors_lmmse= 0;  % Method 4

fprintf('Starting Simulation with %d trials...\n', num_trials);
fprintf('Target SNR: %d dB\n', SNR_dB);

tic;

%% 2. Main Simulation Loop
for trial = 1:num_trials
    
    % --- A. Data Generation ---
    % 1. Sparse Matrix A
    A = sprandn(M, N, sparsity_density);
    A = abs(A); % Non-negative matrix assumption
    A2 = A.^2;
    
    % 2. Signal x (Exponential)
    lambda = 1 + 9 * rand(N, 1); 
    mu_x = 1 ./ lambda;       % E[x]
    var_x = 1 ./ lambda.^2;   % Var(x)
    
    x_true = exprnd(mu_x);
    
    % 3. Observation (AWGN with SNR Control)
    z_true = A * x_true;
    
    % Calculate Signal Power (Average Second Moment)
    P_sig = mean(z_true.^2);
    if P_sig < 1e-10, P_sig = 1e-10; end % Protect against zero signal
    
    % Calculate Target Noise Variance based on SNR
    % SNR = 10*log10(P_sig / sigma2)
    sigma2 = P_sig * 10^(-SNR_dB/10);
    
    % Generate Gaussian Noise
    w = sqrt(sigma2) * randn(M, 1);
    y = z_true + w;
    
    [~, true_max_idx] = max(x_true);
    
    % --- B. Run Sum-Product GAMP (Methods 1, 2, 3) ---
    x_hat = mu_x;     % Init with prior mean
    vx = var_x;       % Init with prior var
    s_hat = zeros(M, 1);
    
    % Variables to store converged state
    r_final = zeros(N, 1);
    vr_final = zeros(N, 1);
    
    for t = 1:max_iter
        % Output Linear
        vp = A2 * vx;
        z_hat = A * x_hat - vp .* s_hat;
        
        % Output Nonlinear (Standard AWGN)
        % Note: sigma2 is now the dynamically calculated value
        vs = 1 ./ (vp + sigma2);
        s_hat_new = (y - z_hat) ./ (vp + sigma2);
        
        s_hat = damp * s_hat_new + (1-damp) * s_hat;
        
        % Input Linear
        vr = 1 ./ (A2' * vs);
        vr = max(real(vr), 1e-10); % Safety
        
        r_hat = x_hat + vr .* (A' * s_hat);
        
        % Input Nonlinear (Exponential MMSE)
        [x_hat_new, vx_new] = estim_exponential(r_hat, vr, lambda);
        
        % Update
        x_hat = damp * x_hat_new + (1-damp) * x_hat;
        vx = damp * vx_new + (1-damp) * vx;
        vx = max(real(vx), 1e-10); % Safety
        
        % Save state
        r_final = r_hat;
        vr_final = vr;
        
        if norm(x_hat - x_hat_new)/norm(x_hat + 1e-9) < tol, break; end
    end
    
    % Method 1: Max of GAMP Mean
    [~, idx_gamp_mean] = max(x_hat);
    
    % Method 2: Max of GAMP Probability
    mu_eff = r_final - lambda .* vr_final;
    sigma_eff = sqrt(vr_final);
    
    alpha = -mu_eff ./ sigma_eff;
    Phi_alpha = normcdf(alpha);
    Z_valid = max(1 - Phi_alpha, 1e-10);
    
    % Vectorized MC Sampling
    U = rand(N, num_samples_mc);
    P_target = bsxfun(@plus, Phi_alpha, bsxfun(@times, U, Z_valid));
    
    % Safety Clamp for P_target
    P_target = min(max(P_target, 1e-10), 1-1e-10);
    
    X_samples = mu_eff + sigma_eff .* norminv(P_target);
    
    [~, max_indices] = max(X_samples, [], 1);
    idx_gamp_prob = mode(max_indices);
    
    % Method 3: Max of GAMP Mode (MAP)
    % MAP estimator: max(0, r - lambda*v)
    x_map = max(0, r_final - lambda .* vr_final);
    [~, idx_gamp_map] = max(x_map);
    
    % --- C. Method 4: LMMSE Estimator ---
    A_full = full(A);
    Cx = diag(var_x);
    Cw = sigma2 * eye(M); % Use the calculated sigma2
    
    mean_y = A_full * mu_x; % Noise is zero-mean for AWGN
    Cyy = A_full * Cx * A_full' + Cw;
    Cxy = Cx * A_full';
    
    % Regularized Solve
    x_lmmse = mu_x + Cxy * ((Cyy + 1e-10*eye(M)) \ (y - mean_y));
    
    [~, idx_lmmse] = max(x_lmmse);
    
    % --- D. Check Mismatch ---
    if idx_gamp_mean ~= true_max_idx, errors_mean = errors_mean + 1; end
    if idx_gamp_prob ~= true_max_idx, errors_prob = errors_prob + 1; end
    if idx_gamp_map  ~= true_max_idx, errors_map  = errors_map + 1; end
    if idx_lmmse     ~= true_max_idx, errors_lmmse= errors_lmmse + 1; end
    
    if mod(trial, num_trials/10) == 0
        fprintf('Completed %d / %d trials...\n', trial, num_trials);
    end
end
elapsed_time = toc;

%% 3. Results Comparison
pm_mean = errors_mean / num_trials;
pm_prob = errors_prob / num_trials;
pm_map  = errors_map  / num_trials;
pm_lmmse= errors_lmmse/ num_trials;

fprintf('\n========== SIMULATION RESULTS ==========\n');
fprintf('Total Trials: %d\n', num_trials);
fprintf('SNR: %d dB\n', SNR_dB);
fprintf('N = %d, M = %d, Density = %.2f\n', N, M, sparsity_density);
fprintf('Noise Type: Additive White Gaussian Noise\n');
fprintf('----------------------------------------\n');
fprintf('Method 1 (GAMP Mean)   Mismatch Rate : %.2f%%\n', pm_mean*100);
fprintf('Method 2 (GAMP Prob)   Mismatch Rate : %.2f%%\n', pm_prob*100);
fprintf('Method 3 (GAMP MAP)    Mismatch Rate : %.2f%%\n', pm_map*100);
fprintf('Method 4 (LMMSE)       Mismatch Rate : %.2f%%\n', pm_lmmse*100);
fprintf('----------------------------------------\n');

% Relative improvements
fprintf('Improvement GAMP Mean over Prob  : %.2f%%\n', (pm_prob - pm_mean)/pm_prob*100);
fprintf('Improvement GAMP Mean over MAP   : %.2f%%\n', (pm_map - pm_mean)/pm_map*100);
fprintf('Improvement GAMP Mean over LMMSE : %.2f%%\n', (pm_lmmse - pm_mean)/pm_lmmse*100);

%% --- Helper Function ---
function [x_post, v_post] = estim_exponential(r, v_r, lam)
    % Safety: Ensure v_r is positive real
    v_r = max(real(v_r), 1e-10);
    r = real(r);
    
    mu_shifted = r - lam .* v_r;
    sigma = sqrt(v_r);
    alpha = -mu_shifted ./ sigma;
    
    h = zeros(size(alpha));
    
    % Threshold for numerical stability
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
    
    x_post = mu_shifted + sigma .* h;
    v_post = v_r .* (1 - h .* (alpha + h));
    
    x_post = max(x_post, 0);
    v_post = max(v_post, 1e-10); 
end
