%%
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
diff_threshold = [5,10,2,15,2,2,3,15,0.25,0.125,0.16,0.14];
len =length(values);
outliers_indices = cell(len,1);
outliers_times = cell(len,1);
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
%% find outliers, outliers could be events.
for i=1:len
    outliers_indices{i} = find(isoutlier(values{i})==1);
    a = outliers_indices{i};
    outliers_times{i} = times{i}(a);
end
%% filter outliers
% if the distance between two outliers are less than 1 minute, than dismiss
% the second outlier
for i = 9:12
   [outliers_indices{i},outliers_times{i}] = filterOutliers(outliers_indices{i},outliers_times{i});
end

%% plot outliers
for i=1:len
    figure
    plot(times{i},values{i},times{i}(outliers_indices{i}),values{i}(outliers_indices{i}),'x');
end
%% calculate events
event_values = coincidence(times,values,outliers_times,60);
event_times = outliers_times;

%% filter events, if events values < 3, discard them.
% after filter the events, we will get a new_event_times,new_event_valus
% and new_outliers_indices.
[ new_event_times,new_event_values,new_outliers_indices] = filterEventsByWeight(times,...
event_times, event_values, outliers_indices, 3 );
%% plot events
for i = 1:len
    figure
    hold('on');
    fig1 = plot(times{i},values{i});
    fig2 = plot(new_event_times{i},values{i}(new_outliers_indices{i}),'x');
    title(names{i});
    xlabel("time");
    ylabel(ylabels(i));
    if length(fig2)~=0
        legend([fig1,fig2],"data","events");
    else
        legend([fig1],"data");
    end
    hold('off');
    %saveas(gcf,['eventDetection' num2str(i) '.png']);
end
