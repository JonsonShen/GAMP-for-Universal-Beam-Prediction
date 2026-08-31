function x = generate_tx_para(M, N,n_aop, n_zop, range_aop, range_zop, center_aop,center_zop)
%x(i): (M, N, AoPeak, ZoPeak)
    n_beams = n_aop*n_zop;
    temp = linspace(center_aop-range_aop,center_aop+range_aop,2*n_aop+1);
    aop = temp(2:2:end);
    temp = linspace(center_zop-range_zop,center_zop+range_zop,2*n_zop+1);
    zop = temp(2:2:end);
    [temp_x, temp_y] =meshgrid(aop,zop);
    x = [M*ones(n_beams,1) N*ones(n_beams,1) temp_x(:) temp_y(:) ];
end
