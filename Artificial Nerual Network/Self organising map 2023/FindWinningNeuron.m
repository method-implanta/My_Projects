function winningNeuron = FindWinningNeuron(weightMatrix,dataInput)

winningNeuronValue = transpose(squeeze(weightMatrix(1,1,:)));
winningNeuron = [1 1];

for i = 1:40
    for j = 1:40
        if CalculateVectorAngle(winningNeuronValue,dataInput) > CalculateVectorAngle(transpose(squeeze(weightMatrix(i,j,:))),dataInput)
            winningNeuronValue = transpose(squeeze(weightMatrix(i,j,:)));
            winningNeuron = [i j];
        end
    end
end

end