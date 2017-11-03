for i = 1:len
    figure
    hold('on');
    fig1 = plot(times{i},values{i});
    fig2 = plot(new_event_times{i},values{i}(new_outliers_indices{i}),'x');
    title(names{i});
    xlabel("time");
    ylabel("value");
    legend([fig1,fig2],"111","222");
    hold('off');
    %saveas(gcf,['eventDetection' num2str(i) '.png']);
end