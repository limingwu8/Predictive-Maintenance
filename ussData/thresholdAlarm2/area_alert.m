function [ averages ] = area_alert(X,Y,window_size,stride,upper_limit )
% calculate area, moving window = 1 hour
clear averages;

%moving_average = tsmovavg(Y','s',window_size);

i = 0;
left_time = X(1) + i*stride;
right_time = left_time + seconds(window_size);
while right_time <= X(length(X))
   i = i+1;
   disp(i);
   averages(i) = average_alarm(X,Y,left_time,right_time);
   left_time = left_time + seconds(i*stride);
   right_time = left_time + seconds(window_size);
end
averages = fillmissing(averages,'previous');
%[m,ind] = max(averages);
[m,ind] = find(averages>=upper_limit);
% ind = 1+(ind-1)*stride;
figure
hold on
p1 = plot(X,Y);
%p2 = plot(X,moving_average);
p3 = 0;
j = 1;
window_left_j = X(1) + seconds(ind(j)*stride);
window_right_j = window_left_j + seconds(window_size);
while (~isempty(ind) & j <= length(ind) & window_right_j<= X(length(X)))
    idx = find(X>=window_left_j & X<=window_right_j);
    abnormal_time = X(idx);
    abnormal_value = Y(idx);
    p3 = plot(abnormal_time,abnormal_value,'x');
    j = j + 1;
end
title('2 OIL RETURN TEMPERATURE');
xlabel('Time');
ylabel('Fahrenheit');
if p3~=0
    legend([p1,p3],'original data','abnormal data');
else
    legend([p1],'original data');
hold off
end



