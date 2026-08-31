a = zeros(8, 1);
b = zeros(8, 1);
%%
k = mean_yn(1)-2;
a(k) = error_lmmse/100000;
b(k) = error_opt/10000;

%%
b = b ./ 10;

%%
figure;
semilogy([3, 4, 5, 6, 7, 8, 9, 10],a,'-o',[3, 4, 5, 6, 7, 8, 9, 10],b,'-hex');
xlabel('Maximun beamforming gain');
ylabel('Prediction error rate');
grid on;
grid minor;
legend('LMMSE','Optimal estimator');
title('Small cluster size prediction error rate comparison');