function CDL_x = load_CDL(csv_file, n_cluster)
    % read CSV file until n cluster
    % skip the first column: Cluster #, second column: Absolute Delay
    % reading from column 3 (Power [dB])
    data = csvread(csv_file, 1, 2);  

    % Extract the necessary columns: Power, AOD, AOA, ZOD, ZOA
    power_dB = data(1:n_cluster, 1);  % Power [dB] (column 3 in the CSV)
    AoD = data(1:n_cluster, 2);       % Azimuth of Departure (AOD)
    AoA = data(1:n_cluster, 3);       % Azimuth of Arrival (AOA)
    AoA = sign(AoA).*(180-abs(AoA));  % rotate coordinate to panel
    ZoD = data(1:n_cluster, 4);       % Zenith of Departure (ZOD)
    ZoA = data(1:n_cluster, 5);       % Zenith of Arrival (ZOA)

    % Convert Power from dB to linear scale
    power_linear = 10 .^ (power_dB / 20); % it's power in dB but we use it for Rayleigh distributed channel coefficient, therefore /20 instead of /10

    % Construct the CDL_x matrix [Power (linear), AoD, AoA, ZoD, ZoA]
    CDL_x = [power_linear, AoD, AoA, ZoD, ZoA];
end
