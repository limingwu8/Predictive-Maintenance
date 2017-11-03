%% the index array is the index of the sensors in data that we need to find events.
index = [1,4,6,7,8,12,16,17,18,19,20,21];
% find out when the above sensors have the following state
eventState = ["Bad","Bad","Bad","Bad","Bad",...
    "Bad","Bad","Bad","Bad","Bad","Bad","Bad"];
events = cell(length(index),3); % format : eventsTime, eventState, sensorName
j = 1;
for i = index
    name = data{i,4};
    time = data{i,1};
    value = data{i,2};
    idx = ismember(value,eventState(j));
    eventsTime = time(idx);
    events{j,1} = eventsTime;
    events{j,2} = eventState(j);
    events{j,3} = name;
    j = j + 1;
end

%% plot events
colors = {[0,0,0,0.5],[0,0,1,0.5],[0,1,0,0.5],[0,1,1,0.5],...
    [1,0,0,0.5],[1,0,1,0.5],[1,1,0,0.5],[1,1,1,0.5],[0.5,0.5,0],...
    [0,0,0.5,0.5],[0,0.5,0,0.5],[0.5,0.5,0,0.5]};
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
        hold(ax{p},'on');
        plt = plot(ax{p},times{j},values{j});
        %ylim auto;
        yl{p} = ylim(ax{p});
        
        pl = line(ax{p},[window_middle window_middle],[yl{p}(1) yl{p}(2)]);
        pl.Color = [1,0,0,0.4];
        pl.LineStyle = '--';
        
        title(ax{p},names{j});
        xlabel(ax{p},"time");
        ylabel(ax{p},ylabels(j));
        %legend([plt,pl],"data","event");
        
        % plot other events
        for t = 1:size(events,1)
            eventsTime = events{t,1};
            color = colors{t};
            for z = 1:size(eventsTime)
               %pl = line(ax{p},[eventsTime(z) eventsTime(z)],[yl{p}(1) yl{p}(2)]); 
               pl = plot(ax{p},eventsTime(z),yl{p}(1) + (yl{p}(2)-yl{p}(1))/2, '*','MarkerSize',10);
               pl.Color = [1,0.647,0];
               %pl.Color = color;
               
            end
        end
        
        hold(ax{p},'off');
    end
    % save figures
%      scrsz = get(0,'ScreenSize');  % Screensize [left, bottom, width, height]
%      set(gcf,'Position',scrsz);   
%      saveas(gcf,['figure' num2str(i) '.png']);
end