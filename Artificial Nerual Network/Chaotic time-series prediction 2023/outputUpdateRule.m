function newOutputs = outputUpdateRule(reservoirNeurons,outputWeights)

newOutputs = zeros(3,1);

for i = 1:3
    newOutputs(i) = sum(reservoirNeurons .* outputWeights(:,i));
end

end