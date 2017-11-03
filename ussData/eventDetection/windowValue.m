% outlier_time : outlier time from one sensor
% left: datetime, the left time of a window
% right: datetime, the right time of a window
% window_value_old: current weight of this window, if it's first time,
% should be [1,1,1,...]
%% given a time window, calculate the total weight of the data from all sensors.
function [ window_value,tf ] = windowValue(left,right,window_value_old,outlier_time)
    if isempty(outlier_time)
        window_value = window_value_old;
        return;
    end
    window_size = seconds(right-left)+1;
    len = max(size(outlier_time));
    window = linspace(left,right,seconds(right-left)+1);
    %window_value = zeros(max(size(window)),1);
    %window_flag = zeros(max(size(window)),1);
    window_value = window_value_old;
    window_value_plus1 = window_value + 1;
    for i=1:len
        win_left = outlier_time(i) - seconds(floor(window_size/2));
        win_right = outlier_time(i) + seconds(floor(window_size/2));
        win = linspace(win_left,win_right,seconds(win_right-win_left)+1);
        intersect_time = intersect(window,win);
        [tf, index] = ismember(intersect_time, window);
        window_value(index) = window_value_plus1(index);
    end
end
