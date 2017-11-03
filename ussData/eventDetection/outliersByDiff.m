function [ outliers_indices ] = outliersByDiff( times, values,diff_threshold )
    outliers_indices = cell(length(times),1);
    for i=1:length(times)
        ind = diff(values{i});
        outliers_indices{i} = find(ind > diff_threshold(i));
    end
end