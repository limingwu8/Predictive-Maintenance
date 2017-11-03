%% load data
y = value_2_OIL_RETURN_TEMPERATURE;
x = time_2_OIL_RETURN_TEMPERATURE;
%% remvoe outliers
y = filloutliers(y,'previous','quartiles');
%% calculate area, moving window = 1 hour
window_size = 3600;
stride = 60; % every time, window move stride in x asix.
upper_limit = 140;
moving_area_hour = area_alert( x,y,window_size,stride,upper_limit);