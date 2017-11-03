figure % new figure
ax1 = subplot(5,1,1); 
ax2 = subplot(5,1,2); 
ax3 = subplot(5,1,3); 
ax4 = subplot(5,1,4); 
ax5 = subplot(5,1,5); 

hold(ax1,'on')
plot(ax1,time_2_tank_filter_out_pressure,value_2_tank_filter_out_pressure)
plot(ax1,time_alarm,0.04,'*')
%plot(ax1,time_alarm(1),0.04,'*')
title(ax1,'2 TANK FILTER OUT PRESSURE')
ylabel(ax1,'')
hold(ax1,'off')

hold(ax2,'on')
%[pks,locs] = findpeaks(value_2_tank_level)
%plot(ax2,time_2_tank_level,value_2_tank_level,time_2_tank_level(locs),pks,'.')
plot(ax2,time_2_tank_level,value_2_tank_level)
plot(ax2,time_alarm,19,'*')
title(ax2,'2 TANK LEVEL')
ylabel(ax2,'')
hold(ax2,'on')

hold(ax3,'on')
%[pks,locs] = findpeaks(value_2_tank_temperature)
%plot(ax3,time_2_tank_temperature,value_2_tank_temperature,time_2_tank_temperature(locs),pks,'.')
plot(ax3,time_2_tank_temperature,value_2_tank_temperature)
plot(ax3,time_alarm,130,'*')
title(ax3,'2 TANK TEMPERATURE')
ylabel(ax3,'')
hold(ax3,'on')

hold(ax4,'on')
%[pks,locs] = findpeaks(value_FT_202B)
%plot(ax4,time_FT_202B,value_FT_202B,time_FT_202B(locs),pks,'.')
plot(ax4,time_FT_202B,value_FT_202B)
plot(ax4,time_alarm,0.5,'*')
title(ax4,'FT 202B')
ylabel(ax4,'')
hold(ax4,'on')

hold(ax5,'on')
%[pks,locs] = findpeaks(value_FT_204B)
%plot(ax5,time_FT_204B,value_FT_204B,time_FT_204B(locs),pks,'.')
plot(ax5,time_FT_204B,value_FT_204B)
plot(ax5,time_alarm,0.3,'*')
title(ax5,'FT 204B')
ylabel(ax5,'')
hold(ax5,'on')

ylim(ax2,[18.4,19.6]) % this line should be put after plot()
ylim(ax3,[118,158])
ylim(ax4,[0,1.6])
ylim(ax5,[0.1,0.7])

