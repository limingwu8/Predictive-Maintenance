function [ eventsInfo ] = coincidence( beginTime,endTime,outliers_times,window_size )

    eventsInfo = cell(0,4);
    window_left = beginTime;
    window_right = window_left + seconds(window_size);
    i = 0;
    while window_right <= endTime
        i = i + 1;
        sensor_idx = [];
        for j = 1:length(outliers_times)
            outlierTime = outliers_times{j}; % vector, outlier time of sensor i
            idx = find(outlierTime>=window_left & outlierTime<window_right);
            if ~isempty(idx)
                sensor_idx(end+1) = j;
            end
        end
        
        eventsInfo(i,:) = {window_left,window_right,length(sensor_idx),sensor_idx};
        
        window_left = window_right;
        window_right = window_left + seconds(window_size);
    end
end

