function gain = BM_gain(params, Z, A)
    % params: tx_params, rx_params in generate_y (M, N, AoPeak, ZoPeak)
    % A: Azimuth angle in degree
    % Z: Zenith angle in degree

    % M number of beam on M axis
    % N number of beam on N   axis
    % ppM_deg : phase progression from element to element in M axis, horizontal, y
    % ppN_deg : phase progression from element to element in N axis, vertical, z

   

    
    M = params(1);  % horizontal element number
    N = params(2);  % vertical element number
    AoPeak = params(3);  % Azimuth peak direction
    ZoPeak = params(4);  % Zenith peak direction
    sep = 1/2; % fraction of wavelength
    ppM_deg = sep * 360 * sind(AoPeak);  % Phase progression along the azimuth direction (M axis)
    ppN_deg = sep * 360 * cosd(ZoPeak) / sqrt(1 - (sind(ZoPeak))^2 * (sind(AoPeak))^2);  % Phase progression along the zenith direction (N axis)


    % Parameters for Gain_per_element.m (ex. from BM_Tx_gain_main.m )
    phi_3dB = 90; % 260
    theta_3dB = 90; % 130
    A_m = 30;
    SLA_v = 30;
    G_E_max = 5.5;

    
    % Calculate per element gain using Gain_per_element.m
    Gain_element = Gain_per_element(phi_3dB, theta_3dB, A_m, SLA_v, G_E_max, Z, A);
    
    % Calculate the total beamforming gain using Gain_BM.m (linear)
    gain= Gain_BM(M, N, Gain_element, ppM_deg, ppN_deg, Z, A)/(M*N);

    %linear gain from dB
    %gain = 10.^(gain_dB/10);
    
end
