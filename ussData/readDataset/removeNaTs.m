for i = 1:length(data)
   time = data{i,1};
   value = data{i,2};
   disp(['sensor: ' num2str(i)]);
   disp([length(time) length(value)]);
   if isempty(value)
       continue;
   end
   natIdx = find(isnat(time));
   disp('nat index:');
   disp(natIdx);
   time(natIdx) = [];
   value(natIdx) = [];
   disp([length(time) length(value)]);
   data{i,1} = time;
   data{i,2} = value;
end