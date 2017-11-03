% outlier_times: datetime, at which time there are outliers
% outlier_indices: integer, indices of outlier_times in times
%% if the distance between two outlier times are less than 1 minute, dismiss the second outlier.
function [ new_outliers_indices,new_outliers_times ] = filterOutliers(outlier_indices,outlier_times)
    len = length(outlier_times);
    interval = seconds(60*5);
    for i = 1:len-1
        if outlier_times(i+1)-outlier_times(i)<interval
            outlier_indices(i+1) = nan;
        end
    end
    nan_indices = find(isnan(outlier_indices)==1);
    outlier_indices(nan_indices) = [];
    outlier_times(nan_indices) = [];
    new_outliers_indices = outlier_indices;
    new_outliers_times = outlier_times;
end

