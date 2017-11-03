i=1
sensor_idx = cell2mat(updatedEventsInfo(i,4));  % e.g. 1,3,9
window_left = updatedEventsInfo{i,1};
%window_left_idx = find
window_right = updatedEventsInfo{i,2};
window_middle = window_left + (window_right - window_left)/2;
% plot correlated sensors, e.g. 1,3,9. plot them in one figure, and
% fill the time window to red 
figure
ax1 = subplot(3,1,1);
ax2 = subplot(3,1,2);
ax3 = subplot(3,1,3);
yl = cell(0,1);
p = 0;
j = 3
p = p+1;
hold('on');
plt = plot(ax1,1,1,'x');
%plot(ax{p},datetime(2017,07,07,0,0,0),0.05)
scatter(ax1,1,2,'o')
%ylim auto;
yl{p} = ylim(ax{p});


% plot other events
for t = 1:size(events,1)
    eventsTime = events{t,1};
    color = colors{t};
    for z = 1:size(eventsTime)
       %pl = line(ax{p},[eventsTime(z) eventsTime(z)],[yl{p}(1) yl{p}(2)]); 
       plot(ax{p},eventsTime(z),yl{p}(1) + (yl{p}(2)-yl{p}(1))/2, 'x');
       %pl.Color = color;

    end
end
hold('off');
