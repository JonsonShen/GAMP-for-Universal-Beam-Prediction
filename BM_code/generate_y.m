function x = generate_y(CDL_x, tx_params, rx_params, n_samples,snr_dB)
    % CDL_x: n_cluster x 5 matrix, columns: power(linear), AoD, AoA, ZoD, ZoA
    % tx_params, rx_params: M, N, AoPeak, ZoPeak
    % y_samples: output signal samples
    
    n_beams = length(tx_params(:,1));
    y_samples = zeros(n_samples, n_beams);
    gtx = 0;
    grx = 0;
    
    for i_sample = 1:n_samples
        AoD = CDL_x(:, 2);    
        AoA = CDL_x(:, 3);   
        ZoD = CDL_x(:, 4);    
        ZoA = CDL_x(:, 5);
        % Rx beamforming gain        
        grx = BM_gain(rx_params, ZoA, AoA);
        % Rayleigh fading channel gain with using linear power
        sig_std = sqrt(sum(CDL_x(:, 1).^2));
        H = diag(raylrnd(CDL_x(:, 1)).^2);
        % Tx beamforming gain
        for i_tx = 1:n_beams
            gtx = BM_gain(tx_params(i_tx,:), ZoD, AoD);
            y_samples(i_sample,i_tx) = max(gtx' * H * grx+normrnd(0,sig_std/(10^(snr_dB/10))),0);
            %y_samples(i_sample,i_tx) = gtx' * H * grx+normrnd(0,sig_std/(10^(snr_dB/10)));
        end


    end    
    x = y_samples; 
end