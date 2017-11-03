for i = 1:length(t)
   value = t{i,2};
   if isfloat(value{1})
       value = cell2mat(value);
       t(i,2) = {value};
   elseif ischar(value{1})
       value = string(value);
       t(i,2) = {value};
   else
       disp("something wrong");
   end
end