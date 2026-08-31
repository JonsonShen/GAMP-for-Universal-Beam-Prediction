function x = Gain_per_element(phi_3dB, theta_3dB, A_m, SLA_v, G_E_max, theta, phi)
x = zeros(length(theta),1);
A_E_V = zeros(length(theta),1);
%% build E-field [sean] E-field per antenna element(gain)

for iloop = 1:length(theta)
    theta_tmp = theta(iloop);
    A_E_V(iloop) = -min(12*((theta_tmp-90)/theta_3dB)^2, SLA_v);
    %for jloop = 1:length(phi)
        phi_tmp = phi(iloop);
        A_E_H(iloop)= -min(12*((phi_tmp)/phi_3dB)^2,A_m);
        %A_E_H(iloop)= -min(12*((180-abs(phi_tmp))/phi_3dB)^2,A_m);
        x(iloop)=G_E_max-min(A_m,-(A_E_H(iloop)+A_E_V(iloop)));
    %end
end
