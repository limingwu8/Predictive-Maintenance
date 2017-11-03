% time: the time of sensor data
% value: the value of sensor data
% window_size: the window size of the moving window, unit:seconds
% stride: every steps the window moves, unit:seconds
function [ averages ] = movingAverage( time, value, window_size, stride )
    clear averages;
    len = max(size(time));
    %averages = nan(len,1);
    averages = cell(seconds(time(len)-time(1)),3);
    left = time(1);
    right = time(1) + seconds(window_size);
    i = 1;
    while right < time(len)
        averages(i,:) = {left,right,windowAverage(time,value,left,right)};
        left = left + seconds(stride);
        right = right + seconds(stride);
        i = i + 1;
    end
    averages = averages(1:i-1,:);
end

