function [ strength ] = corrStrength( corr )
    if (abs(corr) > 0.5 & abs(corr) <=1)
        strength = 3;
    elseif (abs(corr) > 0.3 & abs(corr) <= 0.5)
        strength = 2;
    elseif (abs(corr) > 0.1 & abs(corr) <= 0.3)
        strength = 1;
    else
        strength = 0;
    end
end

