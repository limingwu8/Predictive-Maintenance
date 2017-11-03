function [ moving_averages ] = moving_average( values,window_size )
    moving_averages = cell(length(values),1);
    for i=1:length(values)
        moving_averages{i} = tsmovavg(values{i}','s',window_size);
    end
end