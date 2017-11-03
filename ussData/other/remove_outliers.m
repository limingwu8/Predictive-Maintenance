function [ new_time,new_data ] = remove_outliers( time,data )
%REMOVE_OUTLIERS
%   remove the outliers and construct a new array
%   data : [1,m]
    [m,n] = size(data);
    if m>n
        data = data';
    end
    [p,q] = size(time);
    if p>q
        time = time';
    end
    
    TF = isoutlier(data,'quartiles');
   
    data(find(TF==1)) = [];
    time(find(TF==1)) = [];
    
    if m>n
        new_data = data';
    else
        new_data = data;
    end
    if p>q
        new_time = time';
    else
        new_time = time;
    end
end

