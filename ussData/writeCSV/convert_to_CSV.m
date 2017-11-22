%% concatenate data from different month
data_name = {'data_0517','data_0617','data_0717','data_0817','data_0917'};
num_value_sensors = 12; % totally 12 sensors have value
concatenated_data = cell(num_value_sensors,3); % [time, value, name]
for i = 1:length(data_name)
   load(data_name{i});
   data = data(cell2mat(data(:,6))==1,:);
   for j = 1:length(data)
       if i == 1
           concatenated_data(j,1:3) = data(j,1:3);
       else
           time1 = concatenated_data{j,1};
           value1 = concatenated_data{j,2};
           time2 = data{j,1};
           value2 = data{j,2};
           concatenated_data{j,1} = [time1;time2];
           concatenated_data{j,2} = [value1;value2];
       end
   end
end
%% resample the data
interval = 3600*6;    % resample time interval
for i = 1:length(concatenated_data)
    time = concatenated_data{i,1};
    value = concatenated_data{i,2};
    [time,value] = sample(time,value,interval);
    concatenated_data{i,1} = time;
    concatenated_data{i,2} = value;
end

%% write to CSV
for i = 1:length(concatenated_data)
    time = concatenated_data{i,1};
    value = concatenated_data{i,2};
    name = concatenated_data{i,3};
    file = fopen([name '.csv'],'w');
    fprintf(file,'%s,%s\n','Time',name);
    for j = 1:length(time)
        fprintf(file,'%s,%.2f\n',datestr(time(j,:),'dd-mmm-yyyy HH:MM:SS'),value(j));
    end
    fclose(file);
end
