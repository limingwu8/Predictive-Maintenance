% [num,txt] = xlsread('FCE_DATA_17SEP_MAR_With_Tag_Attributes.xlsx',3,'BD:BK');

% times = txt(:,1:2:61);
% times([1,2],:) = [];

% process value
% lens = num(1,1:2:61); % length of each sensors
% values = txt(:,2:2:61);
% values([1,2],:) = [];
% num(:,2:2:61) = [];
% txt(:,1:2:61) = [];
% txt([1,2],:) = [];
% num(1,:) = [];

% data = cell(length(lens),2);
% 
% for i = 1:length(lens) %31
%     len = lens(i);
%     time = times(1:len,i);
%     time = string(time);
%     time = datetime(time);
%     
%     if isnan(num(1,i))
%         value = txt(1:len,i);
%         value = string(value);
%     else
%         value = num(1:len,i);
%         nanIndex = find(isnan(value));
%         value(nanIndex) = [];
%         len = len - length(nanIndex);
%         time(nanIndex) = [];
%     end
%     
%     data(i,1) = {time};
%     data(i,2) = {value};
% end

%%
for i = 1:length(lens) %31
    len = lens(i);
    time = times(1:len,i);
    time = dateStringFormat(time,lens);
    time = datetime(time);
    data(i,1) = {time};
end