%%
tic
left = datetime(2017,07,05,0,0,0);
right = datetime(2017,07,06,0,0,0);


average = windowAverage(time_2_OIL_RETURN_TEMPERATURE,...
                        value_2_OIL_RETURN_TEMPERATURE,left,right);

toc