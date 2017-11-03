%% filter events, if events values < weight, discard them.
% after filter the events, we will get a new_event_times,new_event_valus
% and new_outliers_indices.
function [ new_event_times,new_event_values,new_outliers_indices] = filterEventsByWeight(times, event_times, event_values, outliers_indices, weight )
    new_event_times = cell(length(event_times),1);
    new_event_values = cell(length(event_values),1);
    new_outliers_indices = cell(length(outliers_indices),1);
    for i = 1:length(event_times)
       event_time = event_times{i};
       event_value = event_values{i};
       lessthan3_idx = find(event_value<weight);
       event_time(lessthan3_idx) = [];
       event_value(lessthan3_idx) = [];
       new_event_times{i} = event_time;
       new_event_values{i} = event_value;
       [Lia,Locb] = ismember(new_event_times{i},times{i});
       new_outliers_indices{i} = Locb;
    end
end

