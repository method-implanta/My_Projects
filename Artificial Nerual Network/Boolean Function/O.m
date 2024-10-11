function output = O(x, weightVector, thresholdValue)

if (x * weightVector - thresholdValue) == 0
    output = 1;
else
    output = sign(x * weightVector - thresholdValue);
end

end