%%
data1 = data(find(cell2mat(data(:,6))==1),:);
lineWidth = 1.5;
fontSize = 18;
for i = 1:length(data1)
    x = data1{i,1};
    y = data1{i,2};
    % fill values every seconds
    X = linspace(x(1),x(max(size(x))),seconds(x(max(size(x)))-x(1))+1)';
    Y = NaN(1,max(size(X)))';
    [tf, index] = ismember(x, X);
    Y(index) = y;
    Y = fillmissing(Y,'previous');
    
    TITLE = data1{i,4};
    YLABEL = data1{i,5};
    XLABEL = 'Time';
    NAME = data1{i,3};
    
    figure()
    hold('on')
    plot(X,Y,'LineWidth',lineWidth)
    title(TITLE,'FontSize',fontSize)
    xlabel(XLABEL,'FontSize',fontSize)
    ylabel(YLABEL,'FontSize',fontSize)
    hold('off')
    
    scrsz = get(0,'ScreenSize');  % Screensize [left, bottom, width, height]
    set(gcf,'Position',scrsz);   
    saveas(gcf,[NAME num2str(i) '.png']);
end