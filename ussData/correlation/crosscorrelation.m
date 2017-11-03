
%data = list(value_2_MAIN_FILTER_IN_PRESSURE,value_2_MAIN_FILTER_OIL_TEMP)
%% all sensor values and sensor name
all_data = {value_2_MAIN_FILTER_IN_PRESSURE,
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
name = {'2 MAIN FILTER IN PRESSURE',
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
%%fill data
for i=1:max(size(all_data))
    x = all_data(i);
    X = linspace(x(1),x(max(size(x))),seconds(x(max(size(x)))-x(1))+1)';
    Y = NaN(1,max(size(X)))';
    [tf, index] = ismember(x, X);
    Y(index) = y;
    Y = fillmissing(Y,'previous');
end
%% choose 2 sensors from 12 sensors, the possible combination is 12x11/2
% use a array to save the information of cross correlation
total_length = max(size(name))*(max(size(name))-1)/2;
info1 = cell(total_length,1);
info2 = cell(total_length,1);
info3 = cell(total_length,1);
info4 = cell(total_length,1);
% correlation strength, 0-3, 
% Strong correlated 3: [-1,-0.5) or (0.5,1]
% Medium correlation 2: [-0.5,-0.3) or (0.3,0.5]
% Slightly correlation 1: [-0.3,-0.1) or (0.1,0.3]
% no correlation 0: [-0.1,0.1]
info5 = cell(total_length,1); 
%%
t = 0; % current loop
for i = 1:max(size(all_data)-1)
   for j = (i+1):(max(size(all_data)))
       t = t + 1;
       str = ['correlation between ',name{i},' and ',name{j}];
       % determine if it's correlated
       ifcorr = 'no';
       [XCF,lags,bounds] = crosscorr(all_data{i},all_data{j});
       [max_xcf,index_max_xcf] = max(XCF);
       [min_xcf,index_min_xcf] = min(XCF);
       
       max_bounds = max(bounds);
       min_bounds = min(bounds);
       %disp([max_xcf,' ',max_bounds,' ',min_xcf,' ',min_bounds])
       %disp(max_xcf)
       if max_xcf>max_bounds || min_xcf<min_bounds
            ifcorr = 'yes';
       end
       %disp(ifcorr)
       info1{t} = str;
       info2{t} = ifcorr;
       info3{t} = [max_xcf,max_bounds,lags(index_max_xcf)];
       info4{t} = [min_xcf,min_bounds,lags(index_min_xcf)];
       info5{t} = corrStrength(max(abs(max_xcf),abs(min_xcf)));
       
       % plot and save figures
       %figure;
       %scrsz = get(0,'ScreenSize');  % Screensize [left, bottom, width, height]
       %set(gcf,'Position',scrsz);   
       %crosscorr(all_data{i},all_data{j});
       %title(str);
       %saveas(gcf,['corr' num2str(t) '.jpg']);
   end
end
info = [info1,info2,info3,info4,info5];
