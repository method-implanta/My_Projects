function newReservoirNeurons = reservoirUpdateRule(inputNeurons,inputWeights,reservoirNeurons,reservoirWeights)

newReservoirNeurons = zeros(500,1);

for i = 1:500
    newReservoirNeurons(i) = tanh(sum(inputNeurons .* inputWeights(:,i)) + sum(reservoirNeurons .* reservoirWeights(:,i)));
end

end