function [ moving_count ] = count_alert(X,Y,window_size,stride,alarm_temp,count_limit)
% calculate area, moving window = 1 hour
clear moving_count;
%moving_average = tsmovavg(Y','s',window_size);
i = 0;
while window_size+1+stride*i<=max(size(Y))
   i = i+1;
   moving_count(i) = sum(Y(1+stride*(i-1):window_size+stride*(i-1))>alarm_temp);
end
   [m,ind] = find(moving_count>=count_limit);
   ind = ind*stride;
figure
hold on
p1 = plot(X,Y);
%p2 = plot(X,moving_average);
p3 = 0;
j = 1;
while (~isempty(ind) & j <= max(size(ind)) & ind(j)+window_size-1<= max(size(Y)))
    p3 = plot(X(ind(j):ind(j)+window_size-1),Y(ind(j):ind(j)+window_size-1),'x');
    j = j + stride;
end
title('2 OIL RETURN TEMPERATURE','FontSize',13);
xlabel('Time','FontSize',13);
ylabel('Fahrenheit','FontSize',13);
if p3~=0
    le = legend([p1,p3],'original data','abnormal data');
    le.FontSize = 13;
else
    le = legend([p1],'original data');
    le.FontSize = 13;
hold off
end