M = csvread('TANK_LEVEL_all.csv', 1,1);
data = M;

% Randomly shuffle the data by chunks
p = 1;
chunks = [];
for i=1:200:length(data)
    if i+200 <= length(data)
        chunks{p} = data(i:i+200)
    end
    p = p+1;
end
index = randperm(length(chunks));
d = [];
for i=1:length(index)
   d = [d, chunks{index(i)}'];
end
% Interpolate data to 270 points which stands for 9 month
%xq = 0:length(data)/270:length(data);
%vq1 = interp1(data, xq);
filt_data = hampel(d, 20);


figure;
hold on;
plot(d,'LineWidth', 2);
plot(filt_data,'LineWidth', 2);
title('L1', 'fontsize',18);
xlabel('Data points', 'fontsize', 14);
ylabel('CM', 'fontsize', 14);
%plot(filt_data);
lgd = legend('Raw sensor data','Sensor output error removed')
lgd.FontSize = 18;