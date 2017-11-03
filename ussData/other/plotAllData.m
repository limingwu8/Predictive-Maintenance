%% Temperature dataset
figure % 
ax1 = subplot(3,1,1); 
ax2 = subplot(3,1,2); 
ax3 = subplot(3,1,3); 

hold(ax1,'on')
plot(ax1,time_2_MAIN_FILTER_OIL_TEMP,value_2_MAIN_FILTER_OIL_TEMP)
plot(ax1,time_alarm,mean(value_2_MAIN_FILTER_OIL_TEMP),'*')
title(ax1,'2 MAIN FILTER OIL TEMP')
ylabel(ax1,'')
hold(ax1,'off')

hold(ax2,'on')
plot(ax2,time_2_TANK_TEMPERATURE,value_2_TANK_TEMPERATURE)
plot(ax2,time_alarm,mean(value_2_TANK_TEMPERATURE),'*')
title(ax2,'2 TANK TEMPERATURE')
ylabel(ax2,'')
hold(ax2,'on')

hold(ax3,'on')
plot(ax3,time_2_OIL_RETURN_TEMPERATUR,value_2_OIL_RETURN_TEMPERATUR)
plot(ax3,time_alarm,mean(value_2_OIL_RETURN_TEMPERATUR),'*')
title(ax3,'2 OIL RETURN TEMPERATUR')
ylabel(ax3,'')
hold(ax3,'on')

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
ylabel(ax4,'')
hold(ax4,'off')

hold(ax5,'on')
plot(ax5,time_2_MAIN_FILTER_OUT_PRESSURE,value_2_MAIN_FILTER_OUT_PRESSURE)
plot(ax5,time_alarm,mean(value_2_MAIN_FILTER_OUT_PRESSURE),'*')
title(ax5,'2 MAIN FILTER OUT PRESSURE')
ylabel(ax5,'')
hold(ax5,'on')

hold(ax6,'on')
plot(ax6,time_2_TANK_FILTER_IN_PRESSURE,value_2_TANK_FILTER_IN_PRESSURE)
plot(ax6,time_alarm,mean(value_2_TANK_FILTER_IN_PRESSURE),'*')
title(ax6,'2 TANK FILTER IN PRESSURE')
ylabel(ax6,'')
hold(ax6,'on')

hold(ax7,'on')
plot(ax7,time_2_TANK_FILTER_OUT_PRESSURE,value_2_TANK_FILTER_OUT_PRESSURE)
plot(ax7,time_alarm,mean(value_2_TANK_FILTER_OUT_PRESSURE),'*')
title(ax7,'2 TANK FILTER OUT PRESSURE')
ylabel(ax7,'')
hold(ax7,'on')

ylim(ax4,[60,90])

%% tank level
figure
ax8 = subplot(1,1,1);

hold(ax8,'on')
plot(ax8,time_2_TANK_LEVEL,value_2_TANK_LEVEL)
plot(ax8,time_alarm,mean(value_2_TANK_LEVEL),'*')
title(ax8,'2 TANK LEVEL')
ylabel(ax8,'')
hold(ax8,'off')

ylim(ax8,[17,21])

%% FT,PT
figure
ax9 = subplot(4,1,1);
ax10 = subplot(4,1,2);
ax11 = subplot(4,1,3);
ax12 = subplot(4,1,4);

plot(ax9,time_FT_202B,value_FT_202B)
title(ax9,'FT 202B','FontSize',13)
xlabel(ax9,"Time",'FontSize',13);
ylabel(ax9,"Mills",'FontSize',13);

plot(ax10,time_PT_203,value_PT_203)
title(ax10,'PT 203','FontSize',13)
xlabel(ax10,"Time",'FontSize',13);
ylabel(ax10,"Mills",'FontSize',13);

plot(ax11,time_FT_204B,value_FT_204B)
title(ax11,'FT 204B','FontSize',13)
xlabel(ax11,"Time",'FontSize',13);
ylabel(ax11,"Mills",'FontSize',13);

plot(ax12,time_PT_204,value_PT_204)
title(ax12,'PT 204','FontSize',13)
xlabel(ax12,"Time",'FontSize',13);
ylabel(ax12,"Mills",'FontSize',13);
