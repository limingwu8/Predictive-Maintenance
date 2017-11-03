function [ output_args ] = plot_info( new_data1,new_data2 )
    figure()
    subplot(3,1,1);
    plot(new_data1{1},new_data1{2});
    subplot(3,1,2);
    plot(new_data2{1},new_data2{2});
    subplot(3,1,3);
    crosscorr(new_data1{2},new_data2{2});
end

