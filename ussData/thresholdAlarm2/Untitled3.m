tic
left = datetime(2017,07,05,0,0,0);
right = datetime(2017,07,06,0,0,0);
y = value_2_OIL_RETURN_TEMPERATURE;
x = time_2_OIL_RETURN_TEMPERATURE;
idx = find(x>=left & x<=right);
x = x(idx);
y = y(idx);
% fill data
X = linspace(x(1),x(max(size(x))),seconds(x(max(size(x)))-x(1))+1)';
Y = NaN(1,max(size(X)))';
[tf, index] = ismember(x, X);
Y(index) = y;
Y = fillmissing(Y,'previous');
avg = mean(Y)
toc