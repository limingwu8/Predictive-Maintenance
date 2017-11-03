function [ normal ] = std_isNormal(time, data,currDate )
%   data : sensor values from one day
%   set a threshold as +2 std and -2 std, if the data points exceed this
%   threshold for more than 3 times , then abnormal.

    normal = 1;
    % get current date data
    [y,m,d] = ymd(currDate);
    today1 = datetime(y,m,d,0,0,0);
    today2 = datetime(y,m,d,23,59,59);
    today_index = find(time>today1 & time<today2);
    todayData = data(today_index);
    n = max(size(todayData)); % length of the data
    avg = mean(todayData);
    std_dev = std(todayData);
    plus2std = avg + 2*std_dev;
    minus2std = avg - 2*std_dev;
    abnormalTimes = numel(find((todayData>plus2std)==1)) + numel(find((todayData<minus2std)==1)); % how many times does the data exceed the std
    if abnormalTimes > 3
        normal = 0;
    end
end

