figure % new figure
ax1 = subplot(5,1,1); 
ax2 = subplot(5,1,2); 
ax3 = subplot(5,1,3); 
ax4 = subplot(5,1,4); 
ax5 = subplot(5,1,5); 


plot(ax1,time_2_tank_filter_out_pressure,value_2_tank_filter_out_pressure)
title(ax1,'2 TANK FILTER OUT PRESSURE')
ylabel(ax1,'')

[pks,locs] = findpeaks(value_2_tank_level)
plot(ax2,time_2_tank_level,value_2_tank_level,time_2_tank_level(locs),pks,'.')
%plot(ax2,time_2_tank_level,value_2_tank_level)
title(ax2,'2 TANK LEVEL')
ylabel(ax2,'')

[pks,locs] = findpeaks(value_2_tank_temperature)
plot(ax3,time_2_tank_temperature,value_2_tank_temperature,time_2_tank_temperature(locs),pks,'.')
%plot(ax3,time_2_tank_temperature,value_2_tank_temperature)
title(ax3,'2 TANK TEMPERATURE')
ylabel(ax3,'')

%[pks,locs] = findpeaks(value_FT_202B)
%plot(ax4,time_FT_202B,value_FT_202B,time_FT_202B(locs),pks,'.')
plot(ax4,time_FT_202B,value_FT_202B)
title(ax4,'FT 202B')
ylabel(ax4,'')

%[pks,locs] = findpeaks(value_FT_204B)
%plot(ax5,time_FT_204B,value_FT_204B,time_FT_204B(locs),pks,'.')
plot(ax5,time_FT_204B,value_FT_204B)
title(ax5,'FT 204B')
ylabel(ax5,'')

ylim(ax2,[18.4,19.6]) % this line should be put after plot()
ylim(ax3,[118,158])
ylim(ax4,[0,1.6])
ylim(ax5,[0.1,0.7])