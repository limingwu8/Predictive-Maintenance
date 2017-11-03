% time is the time string of all sensors
% lens is the length of data in each sensor
% function [ formatedTime ] = dataStringFormat( time,lens )
%     for col = 1:length(lens)
%         time_col = time(1:lens(col),col);
%         idx = strfind(time_col,'M');
%         for row = 1:length(idx)
%            if(isempty(idx{row}))
%                time_col{row} = [time_col{row} ' 00:00:00 AM'];
%            end
%         end
%         formatedTime{col} = string(time_col);
%     end
% end
function [ formatedTime ] = dateStringFormat(time,lens)
    idx = strfind(time,'M');
    for row = 1:length(idx)
       if(isempty(idx{row}))
           time{row} = [time{row} ' 00:00:00 AM'];
       end
    end
    formatedTime = string(time);
end
