
% Load the CDL matrix from the CSV file
% CDL_x = [Power (linear), AoD, AoA, ZoD, ZoA]
csv_file = 'CDL_A.csv';
n_cluster = 5;
CDL_x = load_CDL(csv_file, n_cluster);

%% Parameters set
% the azimuth range (phi) and zenith range (theta)
phi_range = -180:1:180; 
theta_range = -90:1:225;

% tx_params , rx_params : [M, N, AoPeak, ZoPeak]
M = 4;  % horizontal elements
N = 4;  % vertical elements

% Get index of the maximum power level
[~, max_power_index] = max(CDL_x(:, 1)); 
%AoPeak = 0;  % Boresight direction
%ZoPeak = 90;  % Horizontal elevation direction
AoPeak = CDL_x(max_power_index, 2); % AoD of the highest power cluster
ZoPeak = CDL_x(max_power_index, 4); % ZoD of the highest power cluster

params = [M, N, AoPeak, ZoPeak]; 

gains_azimuth = zeros(1, length(phi_range));
gains_zenith = zeros(1, length(theta_range));


%% Loop for beamforming gain calculation
% Loop with azimuth angle
for i = 1:length(phi_range)
    phi = phi_range(i);
    gains_azimuth(i) = BM_gain(params, phi, ZoPeak);
end

% Loop with zenith angle
for i = 1:length(theta_range)
    theta = theta_range(i);
    gains_zenith(i) = BM_gain(params, AoPeak, theta);
end

%% Plot
% Plot the beamforming gain(linear) vs azimuth angle
figure;
plot(phi_range, gains_azimuth, 'LineWidth', 2);
xlabel('Azimuth Angle (°)');
ylabel('Beamforming Gain (linear)');
title(['Beamforming Gain vs Azimuth Angle for Beam with AoPeak = ', num2str(AoPeak),'°']);
grid on;

% Plot the beamforming gain(linear) vs zenith angle
figure;
plot(theta_range, gains_zenith, 'LineWidth', 2);
xlabel('Zenith Angle (°)');
ylabel('Beamforming Gain (linear)');
title(['Beamforming Gain vs Zenith Angle for Beam with ZoPeak = ', num2str(ZoPeak),'°']);
grid on;
