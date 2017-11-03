function [ normal ] = mean_isNormal( time, data, currDate)
%	time: all time data of a sensor
%   data: all value data of a sensor
%   currDate: current date(year,month,day)
%   if the mean of values from today minus the mean of values from the
%   previous day, then abnormal
    normal = 1;
    [y,m,d] = ymd(currDate);
    today1 = datetime(y,m,d,0,0,0);
    today2 = datetime(y,m,d,23,59,59);
    today_index = find(time>today1 & time<today2);
    todayMean = mean(data(today_index));
    yesterday1 = datetime(y,m,d-1,0,0,0);
    yesterday2 = datetime(y,m,d-1,23,59,59);
    yesterday_index = find(time>yesterday1 & time<yesterday2);
    yesterdayMean = mean(data(yesterday_index));
    
    if abs(todayMean-yesterdayMean) > 2
        normal = 0;
    end
end

