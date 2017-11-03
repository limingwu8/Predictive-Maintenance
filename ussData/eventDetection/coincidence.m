function [ event_values ] = coincidence(times,values,outliers_time,window_size)
    len = max(size(times));
    event_values = cell(len,1);

    for i = 1:len
        outlier_time = outliers_time{i};
        event_value = zeros(length(outlier_time),1);
        sensor_ind = cell(length(outlier_time),1);
        for j = 1:length(outlier_time)
            left = outlier_time(j) - seconds(window_size/2);
            right = outlier_time(j) + seconds(window_size/2);
            [event_value(j), sensor_ind{j}] = maxEventValue(left,right,outliers_time);
            
        end
        event_values{i} = event_value;
    end
end