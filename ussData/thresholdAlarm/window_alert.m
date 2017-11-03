%% load data
y = data{22,2};
x = data{22,1};
%plot(x,y);
%% remove outliers
y = filloutliers(y,'previous','quartiles');
%plot(x,y);
%% fill dataset by using nearst neighbor
%y = fillmissing(y,'previous')
% generage time axis, time scale is seconds
X = linspace(x(1),x(max(size(x))),seconds(x(max(size(x)))-x(1))+1)';
Y = NaN(1,max(size(X)))';
[tf, index] = ismember(x, X);
Y(index) = y;
Y = fillmissing(Y,'previous');

%% moving average, window = 1 hour = 3600 seconds
% window_size = 3600;
% moving_average = tsmovavg(Y','s',24*3*window_size);% 
% plot(X,Y,X,moving_average);
%% calculate area, moving window = 1 hour
window_size = 3600;
stride = 60; % every time, window move stride in x asix.
% upper_limit = 143.6;
upper_limit = 130;
moving_area_hour = average_alarm( X,Y,window_size,stride,upper_limit);
%% calculate area, moving window = 1 day
t1=clock;
window_size = 3600 * 24;
stride = 60; % every time, window move stride in x asix.
upper_limit = 139.23;
moving_area_day = average_alarm( X,Y,window_size,stride,upper_limit);
t2=clock;
etime(t2,t1)
%% calculate count, moving window = 1 hour
window_size = 3600;
stride = 60; % every time, window move stride in x asix.
alarm_temp = 143; % if temperature exceeds this value,then count them.
count_limit = 3000; % if temperature exceeds the alarm_temp for more than count_limit, alarm.
moving_count_hour = frequency_alarm(X,Y,window_size,stride,alarm_temp,count_limit);
%% calculate count, moving window = 1 day
window_size = 3600 * 24;
stride = 60; % every time, window move stride in x asix.
alarm_temp = 143; % if temperature exceeds this value,then count them.
count_limit = 21720; % if temperature exceeds the alarm_temp for more than count_limit, alarm.
moving_count_day = frequency_alarm(X,Y,window_size,stride,alarm_temp,count_limit);