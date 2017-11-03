function [ averages ] = average_alarm(X,Y,window_size,stride,upper_limit )
% calculate area, moving window = 1 hour
clear averages;

%moving_average = tsmovavg(Y','s',window_size);

i = 0;
left = 1 + i*stride;
right = left + window_size;
while right <= max(size(Y))
   i = i+1;
   averages(i) = trapz(Y(left:right))/window_size;
   left = 1 + i*stride;
   right = left + window_size;
end

%[m,ind] = max(averages);
[m,ind] = find(averages>=upper_limit);
ind = 1+(ind-1)*stride;
figure
hold on
p1 = plot(X,Y);
%p2 = plot(X,moving_average);
p3 = 0;
j = 1;
while (~isempty(ind) & j <= max(size(ind)) & ind(j)+window_size-1<= max(size(Y)))
    p3 = plot(X(ind(j):ind(j)+window_size-1),Y(ind(j):ind(j)+window_size-1),'x');
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



