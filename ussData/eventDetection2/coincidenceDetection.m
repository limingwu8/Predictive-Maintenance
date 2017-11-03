%%
data1 = data(find(cell2mat(data(:,6))==1),:);
times = data1(:,1);
values = data1(:,2);
names = data1(:,4);
ylabels = data1(:,5);

% sample

for i=1:len
    [X,Y] = sample(times{i},values{i},30);
    times(i) = {X};
    values(i) = {Y};
end

%diff_threshold = [5,8,0.05,10,1.5,0.024,1.1,10,0.20,0.1,0.15,0.12];
len =length(values);
outliers_indices = cell(len,1);
outliers_times = cell(len,1);
%% remove NaT
% for i = 1:len
%    value = values{i};
%    time = times{i};
%    nat_index = find(isnat(time)==1);
%    value(nat_index) = [];
%    time(nat_index) = [];
%    values{i} = value;
%    times{i} = time;
% end
%% find outliers, outliers could be events.
for i=1:len
    outliers_indices{i} = find(isoutlier(values{i})==1);
    a = outliers_indices{i};
    outliers_times{i} = times{i}(a);
end
% filter outliers
% if the distance between two outliers are less than 1 minute, than dismiss
% the second outlier

% for i = 9:12
%    [outliers_indices{i},outliers_times{i}] = filterOutliers(outliers_indices{i},outliers_times{i});
% end
% for i=1:len
%     outliers_indices{i} = find(diff(values{i}) > diff_threshold(i));
%     a = outliers_indices{i};
%     outliers_times{i} = times{i}(a);
% end

%% plot outliers
for i=1:len
    figure
    plot(times{i},values{i},times{i}(outliers_indices{i}),values{i}(outliers_indices{i}),'x');
end
%% calculate events
eventsInfo = coincidence(datetime(2017,08,01,0,0,0),datetime(2017,08,30,0,0,0),outliers_times,60*3);

%% filter events, if during a time window, less than 3 sensors are correlated, then discard them.
c = eventsInfo(:,3);
c = cell2mat(c);
keep = find(c>=3);
updatedEventsInfo = eventsInfo(keep,:);

%% plot events
for i = 1:size(updatedEventsInfo,1)
    sensor_idx = cell2mat(updatedEventsInfo(i,4));  % e.g. 1,3,9
    window_left = updatedEventsInfo{i,1};
    %window_left_idx = find
    window_right = updatedEventsInfo{i,2};
    window_middle = window_left + (window_right - window_left)/2;
    % plot correlated sensors, e.g. 1,3,9. plot them in one figure, and
    % fill the time window to red 
    figure
    ax = {};
    for p = 1:length(sensor_idx)
        ax{p} = subplot(length(sensor_idx),1,p);
    end
    yl = cell(0,1);
    p = 0;
    for j = sensor_idx
        p = p+1;
        hold('on');
        plt = plot(ax{p},times{j},values{j});
        %ylim auto;
        yl{p} = ylim(ax{p});
        
        pl = line(ax{p},[window_middle window_middle],[yl{p}(1) yl{p}(2)]);
        pl.Color = [1,0,0,0.4];
        pl.LineStyle = '--';
        
        title(ax{p},names{j});
        xlabel(ax{p},"time");
        ylabel(ax{p},ylabels(j));
        legend([plt,pl],"data","event");
        hold('off');
    end
    % save figures
%      scrsz = get(0,'ScreenSize');  % Screensize [left, bottom, width, height]
%      set(gcf,'Position',scrsz);   
%      saveas(gcf,['figure' num2str(i) '.png']);
end