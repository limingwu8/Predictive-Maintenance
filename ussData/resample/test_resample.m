x = data{22,1};
y = data{22,2};

figure()
plot(x,y);

X = linspace(x(1),x(max(size(x))),seconds(x(max(size(x)))-x(1))+1)';
Y = NaN(1,max(size(X)))';
[tf, index] = ismember(x, X);
Y(index) = y;
Y = fillmissing(Y,'previous');

X = X(1:n:length(X))
Y = Y(1:n:length(Y))
