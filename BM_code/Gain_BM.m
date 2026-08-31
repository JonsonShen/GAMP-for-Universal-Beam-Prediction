function x = Gain_BM(M, N, Gain_element, ppM_deg, ppN_deg, theta, phi)

%% Build beam for M by N array, with N elements in vertical(z), and M in horizontal (y)
% [sean] adding up element, forming one beam, use different ppN/ppM can for
% different beams, 0/0 is the boresign beam

% M number of beam on M axis
% N number of beam on N   axis
% ppM_deg : phase progression from element to element in M axis, horizontal, y
% ppN_deg : phase progression from element to element in N axis, vertical, z

sep = 1/2; % fraction of wavelength

ppM = ppM_deg*pi/180;
ppN = ppN_deg*pi/180;

FieldStrength_lin = zeros(size(Gain_element));
FieldStrength_element_lin = 10.^(Gain_element/20);

for n = 1:N
    for m = 1:M
        el_location = [0, (m-1)*sep, (n-1)*sep];
        for iloop = 1:length(theta)
            theta_tmp = theta(iloop)*pi/180;
            %for jloop = 1:length(phi)
                phi_tmp = phi(iloop)*pi/180;
                % unit field point vector
                r_hat = [cos(phi_tmp)*sin(theta_tmp)...
                    sin(phi_tmp)*sin(theta_tmp)...
                    cos(theta_tmp)];
                % absolute value of spatial phase advance for element(m,n)
                ph_adv = 2*pi*(dot(r_hat,el_location));
                %add sign of ph_adv

                %for each location in space, net phase of each element is
                net_phase = ph_adv - (m-1)*ppM - (n-1)*ppN;

                % add contribution of element to each location in space

                FieldStrength_lin(iloop)=FieldStrength_lin(iloop) + FieldStrength_element_lin(iloop)*exp(1i*net_phase);
                
            %end
        end
    end

end


x = abs(FieldStrength_lin).^2;


             

