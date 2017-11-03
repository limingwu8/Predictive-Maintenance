figure % new figure
ax6 = subplot(5,1,1); 
ax7 = subplot(5,1,2); 
ax8 = subplot(5,1,3); 
ax9 = subplot(5,1,4); 
ax10 = subplot(5,1,5);

hold(ax6,'on')
plot(ax6,time_2_MAIN_FILTER_IN_PRESSURE,value_2_MAIN_FILTER_IN_PRESSURE)
plot(ax6,time_alarm,mean(value_2_MAIN_FILTER_IN_PRESSURE),'*')
title(ax6,'2 MAIN FILTER IN PRESSURE')
ylabel(ax6,'')
hold(ax6,'off')

hold(ax7,'on')
%[pks,locs] = findpeaks(value_2_tank_level)
%plot(ax7,time_2_tank_level,value_2_tank_level,time_2_tank_level(locs),pks,'.')
plot(ax7,time_2_MAIN_FILTER_OIL_TEMP,value_2_MAIN_FILTER_OIL_TEMP)
plot(ax7,time_alarm,mean(value_2_MAIN_FILTER_OIL_TEMP),'*')
title(ax7,'2 MAIN FILTER OIL TEMP')
ylabel(ax7,'')
hold(ax7,'on')

hold(ax8,'on')
%[pks,locs] = findpeaks(value_2_tank_temperature)
%plot(ax8,time_2_tank_temperature,value_2_tank_temperature,time_2_tank_temperature(locs),pks,'.')
plot(ax8,time_2_MAIN_FILTER_OUT_PRESSURE,value_2_MAIN_FILTER_OUT_PRESSURE)
plot(ax8,time_alarm,mean(value_2_MAIN_FILTER_OUT_PRESSURE),'*')
title(ax8,'2 MAIN FILTER OUT PRESSURE')
ylabel(ax8,'')
hold(ax8,'on')

hold(ax9,'on')
%[pks,locs] = findpeaks(value_2_tank_temperature)
%plot(ax8,time_2_tank_temperature,value_2_tank_temperature,time_2_tank_temperature(locs),pks,'.')
plot(ax9,time_2_OIL_RETURN_TEMPERATURE,value_2_OIL_RETURN_TEMPERATURE)
plot(ax9,time_alarm,mean(value_2_OIL_RETURN_TEMPERATURE),'*')
title(ax9,'2 OIL RETURN TEMPERATUR')
ylabel(ax9,'')
hold(ax9,'on')

hold(ax10,'on')
%[pks,locs] = findpeaks(value_2_tank_temperature)
%plot(ax8,time_2_tank_temperature,value_2_tank_temperature,time_2_tank_temperature(locs),pks,'.')
plot(ax10,time_2_TANK_FILTER_IN_PRESSURE,value_2_TANK_FILTER_IN_PRESSURE)
plot(ax10,time_alarm,mean(value_2_TANK_FILTER_IN_PRESSURE),'*')
title(ax10,'2 TANK FILTER IN PRESSURE')
ylabel(ax10,'')
hold(ax10,'on')

ylim(ax6,[55,85])
ylim(ax9,[115,145])

