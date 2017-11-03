%%
times = {time_2_MAIN_FILTER_IN_PRESSURE,
    time_2_MAIN_FILTER_OIL_TEMP,
    time_2_MAIN_FILTER_OUT_PRESSURE,
    time_2_OIL_RETURN_TEMPERATURE,
    time_2_TANK_FILTER_IN_PRESSURE,
    time_2_TANK_FILTER_OUT_PRESSURE,
    time_2_TANK_LEVEL,
    time_2_TANK_TEMPERATURE,
    time_FT_202B,
    time_FT_204B,
    time_PT_203,
    time_PT_204};
values = {value_2_MAIN_FILTER_IN_PRESSURE,
    value_2_MAIN_FILTER_OIL_TEMP,
    value_2_MAIN_FILTER_OUT_PRESSURE,
    value_2_OIL_RETURN_TEMPERATURE,
    value_2_TANK_FILTER_IN_PRESSURE,
    value_2_TANK_FILTER_OUT_PRESSURE,
    value_2_TANK_LEVEL,
    value_2_TANK_TEMPERATURE,
    value_FT_202B,
    value_FT_204B,
    value_PT_203,
    value_PT_204};
names = {'2 MAIN FILTER IN PRESSURE',
    '2 MAIN FILTER OIL TEMP',
    '2 MAIN FILTER OUT PRESSURE',
    '2 OIL RETURN TEMPERATURE',
    '2 TANK FILTER IN PRESSURE',
    '2 TANK FILTER OUT PRESSURE',
    '2 TANK LEVEL',
    '2 TANK TEMPERATURE',
    'FT 202B',
    'FT 204B',
    'PT 203',
    'PT 204'};
ylabels = {
    "inches WC",
    "Fahrenheit",
    "inches WC",
    "Fahrenheit",
    "inches WC",
    "inches WC",
    "inch",
    "Fahrenheit",
    "Mills",
    "Mills",
    "Mills",
    "Mills"
};
len =length(values);
%% remove NaT
for i = 1:len
   value = values{i};
   time = times{i};
   nat_index = find(isnat(time)==1);
   value(nat_index) = [];
   time(nat_index) = [];
   values{i} = value;
   times{i} = time;
end
%% fill outliers
for i=1:len
    values{i} = filloutliers(values{i},'previous','quartiles');
end

%plot(x,y);
%% fill dataset by using nearst neighbor
%y = fillmissing(y,'previous')
% generage time axis, time scale is seconds
%for i=1:len
%    x = times{i};
%    y = values{i};
%   X = linspace(x(1),x(max(size(x))),seconds(x(max(size(x)))-x(1))+1)';
%    Y = NaN(1,max(size(X)))';
%    [tf, index] = ismember(x, X);
%    Y(index) = y;
%    Y = fillmissing(Y,'previous');
%    times{i} = X;
%    values{i} = Y;
%end

%% plot moving averages
moving_averages = moving_average( values,30 );
for i = 1:len
    figure
    hold('on');
    fig1 = plot(times{i},values{i});
    fig2 = plot(times{i},moving_averages{i},'r');
    title(names{i});
    xlabel("time");
    ylabel(ylabels(i));
    legend([fig1,fig2],"data","moving averages");
    hold('off');
    saveas(gcf,['movingAverage' num2str(i) '.png']);
end