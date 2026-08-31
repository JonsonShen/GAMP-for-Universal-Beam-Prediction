function x = generate_y(CDL_x,tx_params,rx_params,n_samples)
    % CDL_x: n_cluster x 5 matrix, columns: power(linear),AoD, AoA, ZoD, ZoA
    % tx_params, rx_params: M,N,AoPeak,ZoPeak
    % BM_gain(bm_params,A,Z) => TBD
    y_samples = zeros(n_samples,1);
    gtx = 0;
    grx = 0;
    for i_sample = 1:n_samples
        for i_cluster = 1:height(CDL_x) %check if hieght returns #rows
            %gtx =
            %BM_gain(tx_params,CDL_x(i_cluster,2),CDL_x(i_cluster,4));
            %grx =
            %BM_gain(rx_params,CDL_x(i_cluster,3),CDL_x(i_cluster,5));
            h = raylrnd(CDL_x(i_cluster,1));
            y_samples(i_sample)= y_samples(i_sample)+gtx*h*grx;
        end
    end

    x = y_samples;