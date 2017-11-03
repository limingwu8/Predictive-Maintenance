%%
%data1: data information of the first sensor, cell array format, which includes
%                                                           time and value
%data2: data information of the second sensor, cell array format, which includes
%                                                           time and value
%interval: sample interval, e.g. if interval = 5s, then every 5s sample once

% after run this function, the first time point and the last time point of
% the two dataset are the same.
function [ new_data1, new_data2 ] = union_sample( data1, data2, interval)
    time1 = data1{1};
    value1 = data1{2};
    time2 = data2{1};
    value2 = data2{2};
    if time1(1) < time2(1)
        fill_time = time2(1);
        temp_values = value1(find(time1<=time2(1)));
        fill_value = temp_values(end);
        
        index = find(time1>=time1(1) & time1<=time2(1));
        time1(index) = [];
        value1(index) = [];
        
        time1 = [fill_time; time1(1:end)];
        value1 = [fill_value; value1(1:end)];
        
    elseif time1(1) > time2(1)
        fill_time = time1(1);
        temp_values = value2(find(time2<=time1(1)));
        fill_value = temp_values(end);
        
        index = find(time2>=time2(1) & time2<=time1(1));
        time2(index) = [];
        value2(index) = [];
        
        time2 = [fill_time;time2(1:end)];
        value2 = [fill_value;value2(1:end)];
    else
        % do nothing
    end
    
    if time1(end) < time2(end)
        fill_time = time1(end);
        temp_values = value2(find(time2>=time1(end)));
        fill_value = temp_values(1);
        
        index = find(time2>=time1(end));
        time2(index) = [];
        value2(index) = [];
        
        time2 = [time2(1:end);fill_time];
        value2 = [value2(1:end);fill_value];
    elseif time1(end) > time2(end)
        fill_time = time2(end);
        temp_values = value1(find(time1>=time2(end)));
        fill_value = temp_values(1);
        
        index = find(time1>=time2(end));
        time1(index) = [];
        value1(index) = [];
        
        time1 = [time1(1:end);fill_time];
        value1 = [value1(1:end);fill_value];
    else
        % do nothing
    end
    [new_time1,new_value1] = sample(time1,value1,interval);
    [new_time2,new_value2] = sample(time2,value2,interval);
    new_data1(1) = {new_time1};
    new_data1(2) = {new_value1};
    new_data2(1) = {new_time2};
    new_data2(2) = {new_value2};
end

