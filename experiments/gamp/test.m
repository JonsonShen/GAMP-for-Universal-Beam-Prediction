%% Simulation: Mismatch Probability (Exp Signal + Exp Noise + NO Norm)
% Goal: Compare GAMP strategies against standard LMMSE
%       **NO Column Normalization applied to A (Hardest Mode)**
%       **Includes Variance Clipping & Strong Damping for Survival**
clear;
clc;
close all;

%% 1. Simulation Parameters
num_trials = 1000;  % Total trials
N = 50;             
M = 25;             
sparsity_density = 1; % Full density (Very hard without normalization)

% SNR Parameter
SNR_dB = 20;        

% GAMP Settings
max_iter = 500;
tol = 1e-5;
num_samples_mc = 5000; 

% --- SURVIVAL DAMPING PARAMETERS ---
% Since A is not normalized, we must start extremely slow.
step_init = 0.05;    % [MODIFIED] Was 0.5. Start tiny.
step_min  = 0.005;   % [MODIFIED] Allow almost zero steps
step_max  = 0.5;     % [MODIFIED] Cap max step to 0.5 (Safety)
step_dec  = 0.5;     
step_inc  = 1.05;    % Slow acceleration

% Counters
errors_mean = 0; errors_prob = 0; errors_map = 0; errors_lmmse= 0;

% Timing
time_gamp_loop = 0; 
time_method_1 = 0; time_method_2 = 0; time_method_3 = 0; time_method_4 = 0; 

fprintf('Starting Simulation...\n');
fprintf('Density: %.2f\n', sparsity_density);
fprintf('Matrix Normalization: DISABLED (Variance Clipping Active)\n');
fprintf('Noise Type: Additive Exponential Noise\n');

tic; 

%% 2. Main Simulation Loop
for trial = 1:num_trials
    
    % --- A. Data Generation ---
    % 1. Sparse Matrix A (Un-normalized)
    A = sprandn(M, N, sparsity_density);
    A = abs(A); 
    
    % [REMOVED] Column Normalization
    % col_norms = sqrt(sum(A.^2, 1));
    % A = bsxfun(@rdivide, A, col_norms);
    
    A2 = A.^2;
    
    % 2. Signal x (Exponential)
    lambda = 1 + 9 * rand(N, 1); 
    mu_x = 1 ./ lambda;       
    var_x = 1 ./ lambda.^2;   
    x_true = exprnd(mu_x);
    
    % 3. Observation
    z_true = A * x_true;
    P_sig = mean(z_true.^2);
    if P_sig < 1e-10, P_sig = 1e-10; end 
    
    P_noise_target = P_sig * 10^(-SNR_dB/10);
    psi = sqrt(2 / P_noise_target);
    w = exprnd(1/psi, M, 1);
    y = z_true + w;
    
    [~, true_max_idx] = max(x_true);
    
    % --- B. Run Sum-Product GAMP ---
    
    t_gamp_start = tic; 
    
    % Initialization
    x_hat = mu_x;     
    vx = var_x;       
    s_hat = zeros(M, 1);
    
    step = step_init;
    diff_norm_prev = 1e10; 
    
    r_final = zeros(N, 1);
    vr_final = zeros(N, 1);
    
    for t = 1:max_iter
        % 1. Output Linear
        vp = A2 * vx;
        % [ADDED] Variance Clipping (Upper Bound)
        % Prevent vp from exploding due to large A elements
        vp = min(max(real(vp), 1e-10), 1e5); 
        
        z_hat = A * x_hat - vp .* s_hat;
        
        % 2. Output Nonlinear (Exp Noise)
        [z_post, vz_post] = estim_output_exp_noise(y, z_hat, vp, psi);
        
        s_hat_new = (z_post - z_hat) ./ vp;
        vs_new    = (1 - vz_post ./ vp) ./ vp;
        
        % 3. Input Linear
        vr = 1 ./ (A2' * vs_new);
        % [ADDED] Variance Clipping (Upper Bound)
        vr = min(max(real(vr), 1e-10), 1e5);
        
        r_hat = x_hat + vr .* (A' * s_hat_new); 
        
        % 4. Input Nonlinear (Exp Signal)
        [x_hat_new, vx_new] = estim_input_exponential(r_hat, vr, lambda);
        
        % --- Robust Damping Logic ---
        diff_norm = norm(x_hat_new - x_hat);
        
        % Looser divergence check to allow large values
        if t > 1 && (isnan(diff_norm) || diff_norm > 1e8) 
             step = step_min;
             x_hat = mu_x; 
             vx = var_x;
             s_hat = zeros(M,1);
        else
            s_hat = step * s_hat_new + (1-step) * s_hat;
            x_hat = step * x_hat_new + (1-step) * x_hat;
            vx    = step * vx_new    + (1-step) * vx;
            vx    = max(real(vx), 1e-10);
            
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
        
        if diff_norm/norm(x_hat + 1e-9) < tol
            break; 
        end
    end
    
    time_gamp_loop = time_gamp_loop + toc(t_gamp_start); 
    
    % --- Method 1: Mean ---
    t_m1 = tic;
    [~, idx_gamp_mean] = max(x_hat);
    time_method_1 = time_method_1 + toc(t_m1);
    
    % --- Method 2: Prob (MC) ---
    t_m2 = tic;
    mu_eff = r_final - lambda .* vr_final;
    sigma_eff = sqrt(vr_final);
    alpha = -mu_eff ./ sigma_eff;
    Phi_alpha = normcdf(alpha);
    Z_valid = max(1 - Phi_alpha, 1e-10);
    U = rand(N, num_samples_mc);
    P_target = min(max(bsxfun(@plus, Phi_alpha, bsxfun(@times, U, Z_valid)), 1e-10), 1-1e-10);
    X_samples = mu_eff + sigma_eff .* norminv(P_target);
    [~, max_indices] = max(X_samples, [], 1);
    idx_gamp_prob = mode(max_indices);
    time_method_2 = time_method_2 + toc(t_m2);
    
    % --- Method 3: MAP ---
    t_m3 = tic;
    x_map = max(0, r_final - lambda .* vr_final);
    [~, idx_gamp_map] = max(x_map);
    time_method_3 = time_method_3 + toc(t_m3);
    
    % --- Method 4: LMMSE ---
    t_m4 = tic;
    A_full = full(A);
    Cx = diag(var_x);
    noise_var = 1/psi^2; 
    noise_mean = 1/psi;
    Cw = noise_var * eye(M); 
    mean_y = A_full * mu_x + noise_mean; 
    Cyy = A_full * Cx * A_full' + Cw;
    Cxy = Cx * A_full';
    x_lmmse = mu_x + Cxy * ((Cyy + 1e-10*eye(M)) \ (y - mean_y));
    [~, idx_lmmse] = max(x_lmmse);
    time_method_4 = time_method_4 + toc(t_m4);
    
    % --- D. Check Mismatch ---
    if idx_gamp_mean ~= true_max_idx, errors_mean = errors_mean + 1; end
    if idx_gamp_prob ~= true_max_idx, errors_prob = errors_prob + 1; end
    if idx_gamp_map  ~= true_max_idx, errors_map  = errors_map + 1; end
    if idx_lmmse     ~= true_max_idx, errors_lmmse= errors_lmmse + 1; end
    
    if mod(trial, num_trials/10) == 0
        fprintf('Completed %d / %d trials...\n', trial, num_trials);
    end
end
elapsed_total = toc;

%% 3. Results Comparison
pm_mean = errors_mean / num_trials;
pm_prob = errors_prob / num_trials;
pm_map  = errors_map  / num_trials;
pm_lmmse= errors_lmmse/ num_trials;

% Time Calculation 
ms_gamp_loop = (time_gamp_loop / num_trials) * 1000;
ms_m1_rank   = (time_method_1 / num_trials) * 1000;
ms_m2_rank   = (time_method_2 / num_trials) * 1000;
ms_m3_rank   = (time_method_3 / num_trials) * 1000;
ms_m4_total  = (time_method_4 / num_trials) * 1000;
ms_m1_total = ms_gamp_loop + ms_m1_rank;
ms_m2_total = ms_gamp_loop + ms_m2_rank;
ms_m3_total = ms_gamp_loop + ms_m3_rank;

fprintf('\n========== SIMULATION RESULTS ==========\n');
fprintf('Density: %.2f (No Normalization)\n', sparsity_density);
fprintf('Method 1 (GAMP Mean)   Mismatch Rate : %.2f%%\n', pm_mean*100);
fprintf('Method 2 (GAMP Prob)   Mismatch Rate : %.2f%%\n', pm_prob*100);
fprintf('Method 3 (GAMP MAP)    Mismatch Rate : %.2f%%\n', pm_map*100);
fprintf('Method 4 (LMMSE)       Mismatch Rate : %.2f%%\n', pm_lmmse*100);
fprintf('----------------------------------------\n');
fprintf('GAMP Iteration Loop     : %.4f ms\n', ms_gamp_loop);
fprintf('Method 2 (Prob Ranking) : %.4f ms  (Total: %.4f ms)\n', ms_m2_rank, ms_m2_total);
fprintf('Method 4 (LMMSE Total)  : %.4f ms\n', ms_m4_total);

%% --- Helper Functions (Same as before) ---
function [x_post, v_post] = estim_input_exponential(r, v_r, lam)
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
    x_post = mu_shifted + sigma .* h;
    v_post = v_r .* (1 - h .* (alpha + h));
    x_post = max(x_post, 0);
    v_post = max(v_post, 1e-10); 
end

function [z_post, vz_post] = estim_output_exp_noise(y, p_hat, vp, psi)
    vp = max(real(vp), 1e-10);
    p_hat = real(p_hat);
    mu_new = p_hat + psi .* vp;
    sigma = sqrt(vp);
    alpha = (y - mu_new) ./ sigma; 
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
