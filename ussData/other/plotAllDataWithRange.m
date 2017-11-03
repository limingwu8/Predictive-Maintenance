%% TEMPERATURE dataset
figure % 
ax1 = subplot(3,1,1); 
ax2 = subplot(3,1,2); 
ax3 = subplot(3,1,3); 

hold(ax1,'on');
plot(ax1,time_2_MAIN_FILTER_OIL_TEMP,value_2_MAIN_FILTER_OIL_TEMP);
plot(ax1,time_alarm,mean(value_2_MAIN_FILTER_OIL_TEMP),'*');
title(ax1,'2 MAIN FILTER OIL TEMP');
xlabel(ax1,'Time');
ylabel(ax1,'Fahrenheit');
%find range and plot
[low,high] = get_normal_range(time_2_MAIN_FILTER_OIL_TEMP,value_2_MAIN_FILTER_OIL_TEMP);
plot(ax1,time_2_MAIN_FILTER_OIL_TEMP,low,'r',time_2_MAIN_FILTER_OIL_TEMP,high,'r');
hold(ax1,'off')

hold(ax2,'on')
plot(ax2,time_2_TANK_TEMPERATURE,value_2_TANK_TEMPERATURE)
plot(ax2,time_alarm,mean(value_2_TANK_TEMPERATURE),'*')
title(ax2,'2 TANK TEMPERATURE')
xlabel(ax2,'Time');
ylabel(ax2,'Fahrenheit')
%find range and plot
[low,high] = get_normal_range(time_2_TANK_TEMPERATURE,value_2_TANK_TEMPERATURE);
plot(ax2,time_2_TANK_TEMPERATURE,low,'r',time_2_TANK_TEMPERATURE,high,'r');
hold(ax2,'off')

hold(ax3,'on')
plot(ax3,time_2_OIL_RETURN_TEMPERATURE,value_2_OIL_RETURN_TEMPERATURE)
plot(ax3,time_alarm,mean(value_2_OIL_RETURN_TEMPERATURE),'*')
title(ax3,'2 OIL RETURN TEMPERATURE')
xlabel(ax3,'Time');
ylabel(ax3,'Fahrenheit')
%find range and plot
[low,high] = get_normal_range(time_2_OIL_RETURN_TEMPERATURE,value_2_OIL_RETURN_TEMPERATURE);
plot(ax3,time_2_OIL_RETURN_TEMPERATURE,low,'r',time_2_OIL_RETURN_TEMPERATURE,high,'r');
hold(ax3,'off')

ylim([ax1,ax2,ax3],[115,155])

%% Pressure dataset
figure % 
ax4 = subplot(4,1,1); 
ax5 = subplot(4,1,2); 
ax6 = subplot(4,1,3); 
ax7 = subplot(4,1,4);

hold(ax4,'on')
plot(ax4,time_2_MAIN_FILTER_IN_PRESSURE,value_2_MAIN_FILTER_IN_PRESSURE)
plot(ax4,time_alarm,mean(value_2_MAIN_FILTER_IN_PRESSURE),'*')
title(ax4,'2 MAIN FILTER IN PRESSURE')
xlabel(ax4,'Time');
ylabel(ax4,'inches WC')
%find range and plot
[low,high] = get_normal_range(time_2_MAIN_FILTER_IN_PRESSURE,value_2_MAIN_FILTER_IN_PRESSURE);
plot(ax4,time_2_MAIN_FILTER_IN_PRESSURE,low,'r',time_2_MAIN_FILTER_IN_PRESSURE,high,'r');
hold(ax4,'off')

hold(ax5,'on')
plot(ax5,time_2_MAIN_FILTER_OUT_PRESSURE,value_2_MAIN_FILTER_OUT_PRESSURE)
plot(ax5,time_alarm,mean(value_2_MAIN_FILTER_OUT_PRESSURE),'*')
title(ax5,'2 MAIN FILTER OUT PRESSURE')
xlabel(ax5,'Time');
ylabel(ax5,'inches WC')
%find range and plot
[low,high] = get_normal_range(time_2_MAIN_FILTER_OUT_PRESSURE,value_2_MAIN_FILTER_OUT_PRESSURE);
plot(ax5,time_2_MAIN_FILTER_OUT_PRESSURE,low,'r',time_2_MAIN_FILTER_OUT_PRESSURE,high,'r');
hold(ax5,'off')

hold(ax6,'on')
plot(ax6,time_2_TANK_FILTER_IN_PRESSURE,value_2_TANK_FILTER_IN_PRESSURE)
plot(ax6,time_alarm,mean(value_2_TANK_FILTER_IN_PRESSURE),'*')
title(ax6,'2 TANK FILTER IN PRESSURE')
xlabel(ax6,'Time');
ylabel(ax6,'inches WC')
%find range and plot
[low,high] = get_normal_range(time_2_TANK_FILTER_IN_PRESSURE,value_2_TANK_FILTER_IN_PRESSURE);
plot(ax6,time_2_TANK_FILTER_IN_PRESSURE,low,'r',time_2_TANK_FILTER_IN_PRESSURE,high,'r');
hold(ax6,'off')

hold(ax7,'on')
plot(ax7,time_2_TANK_FILTER_OUT_PRESSURE,value_2_TANK_FILTER_OUT_PRESSURE)
plot(ax7,time_alarm,mean(value_2_TANK_FILTER_OUT_PRESSURE),'*')
title(ax7,'2 TANK FILTER OUT PRESSURE')
xlabel(ax7,'Time');
ylabel(ax7,'inches WC')
%find range and plot
[low,high] = get_normal_range(time_2_TANK_FILTER_OUT_PRESSURE,value_2_TANK_FILTER_OUT_PRESSURE);
plot(ax7,time_2_TANK_FILTER_OUT_PRESSURE,low,'r',time_2_TANK_FILTER_OUT_PRESSURE,high,'r');
hold(ax7,'off')

ylim(ax4,[60,90])

%% tank level
figure
ax8 = subplot(1,1,1);

hold(ax8,'on')
plot(ax8,time_2_TANK_LEVEL,value_2_TANK_LEVEL)
plot(ax8,time_alarm,mean(value_2_TANK_LEVEL),'*')
title(ax8,'2 TANK LEVEL')
xlabel(ax8,'Time');
ylabel(ax8,'inch')
%find range and plot
[low,high] = get_normal_range(time_2_TANK_LEVEL,value_2_TANK_LEVEL);
plot(ax8,time_2_TANK_LEVEL,low,'r',time_2_TANK_LEVEL,high,'r');
hold(ax8,'off')

ylim(ax8,[17,21])

%% FT,PT
figure
ax9 = subplot(4,1,1);
ax10 = subplot(4,1,2);
ax11 = subplot(4,1,3);
ax12 = subplot(4,1,4);

hold(ax9,'on')
plot(ax9,time_FT_202B,value_FT_202B)
title(ax9,'FT 202B')
xlabel(ax9,'Time');
ylabel(ax9,'Mills');
%find range and plot
[low,high] = get_normal_range(time_FT_202B,value_FT_202B);
plot(ax9,time_FT_202B,low,'r',time_FT_202B,high,'r');
hold(ax9,'off')


hold(ax10,'on')
plot(ax10,time_PT_203,value_PT_203)
title(ax10,'PT 203')
xlabel(ax10,'Time');
ylabel(ax10,'Mills');
%find range and plot
[low,high] = get_normal_range(time_PT_203,value_PT_203);
plot(ax10,time_PT_203,low,'r',time_PT_203,high,'r');
hold(ax10,'off')

hold(ax11,'on')
plot(ax11,time_FT_204B,value_FT_204B)
title(ax11,'FT 204B')
xlabel(ax11,'Time');
ylabel(ax11,'Mills');
%find range and plot
[low,high] = get_normal_range(time_FT_204B,value_FT_204B);
plot(ax11,time_FT_204B,low,'r',time_FT_204B,high,'r');
hold(ax11,'off')

hold(ax12,'on')
plot(ax12,time_PT_204,value_PT_204)
title(ax12,'PT 204')
xlabel(ax12,'Time');
ylabel(ax12,'Mills');
%find range and plot
[low,high] = get_normal_range(time_PT_204,value_PT_204);
plot(ax12,time_PT_204,low,'r',time_PT_204,high,'r');
hold(ax12,'off')
