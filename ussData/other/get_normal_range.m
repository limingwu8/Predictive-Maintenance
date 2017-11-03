function [ low,high,new_data ] = get_normal_range(time, data )
%GET_NORMAL_RANGE
%   calculate the normal range
    %[new_time,new_data] = remove_outliers(time,data);
    new_time = time;
    %new_data = hampel(data,100);
    new_data = filloutliers(data,'previous','quartiles');
    
    %figure
    %plot(new_time,new_data);
    
    std_dev = std(new_data);
    high_value = mean(new_data) + 2*std_dev;
    low_value = mean(new_data) - 2*std_dev;
    high = zeros(max(size(data)),min(size(data)));
    low = zeros(max(size(data)),min(size(data)));
    high(:) = high_value;
    low(:) = low_value;
end