X = getHealPixNodes(10); % [sean]1200 sampling point around sphere, don't change

Numpoints = size(X,1);

fig = 100;
dB_Loss = 0; % Loss modeled as dB_Loss*sin(2*theta)
dB_Uncertainty = 0.0; %Loss modeled as dB_Uncertainty*randomnumberbetween(0,1)


%% grid initialization: range and granularity
theta = [45:1:135]; % upper bound > 90
phi = [0:1:90];
[theta_grid, phi_grid] = meshgrid(theta,phi);


% NOTE: The and dfinition in this subclause is based on the coordinate
% system in subclause 5.4.4.1 of 3GPP TR 37.840 which says:


% The elevation angle of the signal direction is denoted as theta
% (definied between 0 degree and 180 degree, 90 degree represents perpendicular angle to the 
% array antenna aperture) and the azimuth angle is denoted as phi
% (defined betwee -180 degree and 180 degree).
% [sean] grid on the sphere: using earth coordinate, north pole is 0, equator is 90
% boresight is on equator


% [sean] antenna element parameters
phi_3dB = 90; % 260
theta_3dB = 90; % 130
A_m = 30;
SLA_v = 30;
G_E_max = 5.5;

Gain_element = Gain_per_element(phi_3dB, theta_3dB, A_m, SLA_v, G_E_max, theta, phi);

%% build CDF
H = PP_LatLong2CDF_Loss2(Gain_element,theta,phi+180,dB_Loss,dB_Uncertainty,X,fig);

% find spherical coverage point
% s = 0.85; % 85th%ile
% G_drop = 8; % 8dB drop
% a = find(H.Values > s);
% b = H.BinEdges;
% G_sph = b(a(1));
% G_pk = b(a(end));
% G_decision = G_pk - G_drop;
%
% mask = 0.5*(1+sign(Gain-G_decision));
% figure; mesh(mask)

TRP = 10*log10(sum(10.^(H.Data/10))/Numpoints); % [sean] adding up all gains, integrated efficiency
% to add : power per element : TRP/number of element

% normalize element pattern to get gain
% Gain_element = Gain_element-TRP; % G_E_max definition redundant with normalization

% Gain = Gain_element;
Gain_element_max = max(max(Gain_element));


%% Visualize beam

x = max(0, Gain_element).*sin(theta_grid.*pi/180).*cos(phi_grid.*pi/180);
y = max(0, Gain_element).*sin(theta_grid.*pi/180).*sin(phi_grid.*pi/180);
z = max(0, Gain_element').*cos(theta_grid.*pi/180);


%% Build beam for M by N array, with N elements in vertical (z), and M in horizontal (y)
% [sean] adding up element, forming one beam, use different ppN/ppM can
% form
% different beams, 0/0 is the boresign beam
M = 4; % horizontal
N = 4; % vertical

M_target = 8;
N_target = 8;
sep = 1/2;
Gain = Gain_BM(M, N, Gain_element, 0, 0, theta, phi); %boresight beam pattern

ue_cvrg = 0;

if ue_cvrg
    phi_set = [0];
    sep_set = [0];
    num_phi = length(phi_set);
    num_sep = length(sep_set);
    peak_diff = zeros(num_phi, num_sep);
    peak_mag = zeros(num_phi, num_sep);
    peak_idx = zeros(num_phi, num_sep);

 
    %% plot beam patter with different designated peak direction (as a function of phase progression) and element separation

    for phi_i = 1:num_phi
        for sep_i = 1:num_sep

            sep = sep_set(sep_i); % antenna element separation (unit: wavelength)
            phi_beam_deg = phi_set(phi_i);
            theta_beam_deg = 90;
            ppM_deg = sep*360*sind(phi_beam_deg); %phase progression from element to element in M axis, horizontal, y
            ppN_deg = sep*360*cosd(theta_beam_deg)/sqrt(1-(sind(theta_beam_deg))^2*(sind(phi_beam_deg))^2); % phase progression from element to element in N axis, vertical, z


            Gain = Gain_BM(M, N, Gain_element, ppM_deg, ppN_deg, theta, phi);

            [peaks, peaks_ind] = findpeaks(Gain(11,:));
            [sort_peak, sort_idx] = sort(peaks,'descend');
            peak_diff(phi_i, sep_i) = sort_peak(1)-sort_peak(2); % gain difference between first and second strongest peak
            peak_mag(phi_i, sep_i) = sort_peak(1); % strongest peak gain
            peak_idx(phi_i, sep_i)= peak_ind(sort_idx(1))+phi(1); % peak direction


            % Visualize
            % figure; mesh(Gain)
            figure; plot(phi, Gain(11,:)); title(strcat('Beam direction ', num2str(phi_beam_deg),', seperation = ', num2str(sep),'lambda'));
            ylim([-30,50]); grid on; grid minor;

        end
    end

    Gain_array_max = max(max(Gain));

end

% Visualize
figure; mesh(Gain)
colorbar;
% get rid of gain < 0
caxis([-30,50]); 
zlim([-30,50]);
view([0,0,1]);
