time = times(1:lens(30),30);
% time is already in the cell format
time = datenum(time,'MM-dd-yyyy HH:MM:SS PM');
time = datetime(time,'Format','dd-MM-yyyy HH:MM:SS','convertFrom','datenum');