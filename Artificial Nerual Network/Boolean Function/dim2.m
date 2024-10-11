clear;
x = [0 0 ; 0 1 ; 1 0 ; 1 1];

t = zeros(4,16);
for i = 1:16
    t(:,i) = reshape(str2num(transpose(dec2bin(i-1,4))) ,4,1);
end
t(t==0) = -1;

weightVectorVariance = sqrt(1/2);
weightVectorMean = 0;
learningRate = 0.05;
thresholdValue = 0;
trainingEpoch = 20;
trainingResult = zeros(4,1);
resultNumber = zeros(2,1);

weightVector = weightVectorVariance.*randn(2,1) + weightVectorMean;

for i = 1:16
    for j = 1:trainingEpoch
        for k = 1:4
            output = O(x(k,:), weightVector, thresholdValue);
            deltaWeightVector = transpose(learningRate * (t(k,i) - output ) * x(k,:));
            deltaThresholdValue = -learningRate * (t(k,i) - output );
            weightVector = weightVector + deltaWeightVector;
            thresholdValue = thresholdValue + deltaThresholdValue;
        end
    end
    for l = 1:4
        trainingResult(l) = O(x(l,:), weightVector, thresholdValue);
    end
    if trainingResult == t(:,i)
        resultNumber(1) = resultNumber(1) + 1;
    else
        resultNumber(2) = resultNumber(2) + 1;
    end
end
disp(resultNumber(1));