% time: the time of sensor data
% value: the value of sensor data
% left: left point of the window, datetime
% right: right point of the window, datetime
% return the mean of data points in the time window
function [ average ] = average_alarm(time,value, left, right )
    index = find(time>=left & time<=right);
    if isempty(index)
        average = NaN;
        return;
    end
    x = time(index);
    y = value(index);
    len = length(x);
    difx = seconds(diff(x));
    difx(len,1) = seconds(right-x(len));
    total_value = sum(difx.*y(1:len));
    total_value = total_value + y(1)*seconds(x(1)-left);
    average = total_value/seconds(right-left);
end