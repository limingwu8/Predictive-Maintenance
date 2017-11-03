figure();
hold('on');
title('2 OIL RETURN TEMPERATURE');
plt1 = plot(X,Y);
plt2 = plot(X,moving_average);
xlabel('Time');
ylabel('Fahrenheit');
legend([plt1,plt2],'sensor data','moving average');