names = ["ACTIVATE_PANEL_ALARM",
        "COMBUSTION_FAN_OIL_FLOW",
        "EJECTOR_FAN_OIL_FLOW",
        "FAN_LUBE_ESTOP",
        "FAN_ROOM_OIL_TEMP",
        "FILTER_PUMP_CALL_TO_RUN",
        "FILTER_PUMP_CONTROL_STATE",
        "FILTER_PUMP_STATUS",
        "HEATER_CALL_TO_RUN",
        "HEATER_CONTROL_STATE.HS.FURNACE",
        "MAIN_FILTER_1_CONDITION",
        "MAIN_FILTER_2_CONDITION",
        "MAIN_FILTER_IN_PRESSURE",
        "MAIN_FILTER_OIL_TEMP",
        "MAIN_FILTER_OUT_PRESSURE",
        "MAIN_PUMP_A_CALL_TO_RUN",
        "MAIN_PUMP_A_CONTROL_STATE",
        "MAIN_PUMP_A_STATUS",
        "MAIN_PUMP_B_CALL_TO_RUN",
        "MAIN_PUMP_B_CONTROL_STATE",
        "MAIN_PUMP_B_STATUS",
        "OIL_RETURN_TEMPERATURE",
        "TANK_FILTER_CONDITION",
        "TANK_FILTER_IN_PRESSURE",
        "TANK_FILTER_OUT_PRESSURE",
        "TANK_LEVEL",
        "TANK_TEMPERATURE",
        "FT-202B",
        "FT-204B",
        "PT-203",
        "PT-204.HS"
        ];
data = cell(31,2);
data(:,3) = cellstr(names);
for i=1:size(num,2)
    length = num(1,i);
 
    time = datetime(txt(3:3+length-1,2*i-1));
    value = 0;
    if isempty(txt{3,2*i})
        value = num2cell(num(2:length+2-1,i));
    else
        value = txt(3:3+length-1,2*i);
    end
    data{i,1} = time;
    data{i,2} = value;
end