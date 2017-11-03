%%
% x: datetime array
% y: float array
% interval: sample interval, e.g. if interval = 5s, then every 5s sample once
function [ X,Y ] = sample(x,y,interval)
    X = x(1):seconds(1):x(end);
    X = X';
    Y = NaN(1,length(X))';
    [tf, index] = ismember(x, X);
    Y(index) = y;
    Y = fillmissing(Y,'previous');
    
    % sample
    X = X(1:interval:length(X));
    Y = Y(1:interval:length(Y));
end