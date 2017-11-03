% left: datetime, the left time of a window
% right: datetime, the right time of a window
% outliers_time: cell, all outliers in all sensor data
%% in one sensor, calculate the max weight in the window.
function [ max_window_value,sensor_ind ] = maxEventValue(left,right,outliers_time )
    window_size = seconds(right-left)+1;
    len = max(size(outliers_time));
    win_value = zeros(window_size,1);
    sensor_ind = [];
    t = 0;
    for i = 1:len
        outlier_time = outliers_time{i};
        a = win_value;
        win_value = windowValue(left,right,win_value,outlier_time);
        b = win_value;
        if ~isequal(a,b)
            t = t + 1;
            sensor_ind(t) = i;
        end
    end
    window_value = win_value;
    max_window_value = max(window_value);
end

