figure
hold('on')
p1 = plot(X,Y);
index = find(Y>75 | Y<62.5);
p2 = scatter(X(index),Y(index),'x');
title('2 OIL RETURN TEMPERATURE','FontSize',13);
xlabel('Time','FontSize',13);
ylabel('Fahrenheit','FontSize',13);
le = legend([p1,p2],'original data','abnormal data');
le.FontSize = 13;
hold('off')