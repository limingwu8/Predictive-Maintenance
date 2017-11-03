function [ moving_average ] = moving_average( input_args )
%MOVING_AVERAGE
    moving_average = tsmovavg(input_args,'s',5)
end