%%
t1=clock;

averages = movingAverage(time_2_OIL_RETURN_TEMPERATURE,...
                        value_2_OIL_RETURN_TEMPERATURE,3600*24,3600*24);

t2=clock;
etime(t2,t1)

%%
left = datetime(2017,07,05,0,0,0);
right = datetime(2017,07,06,0,0,0);
t1=clock;

average = windowAverage(time_2_OIL_RETURN_TEMPERATURE,...
                        value_2_OIL_RETURN_TEMPERATURE,left,right);

t2=clock;
etime(t2,t1)
%%
t1 = clock;
windowAverage(time_2_OIL_RETURN_TEMPERATURE,value_2_OIL_RETURN_TEMPERATURE,... 
time_2_OIL_RETURN_TEMPERATURE(1), time_2_OIL_RETURN_TEMPERATURE(670) )
t2 = clock;
etime(t2,t1)