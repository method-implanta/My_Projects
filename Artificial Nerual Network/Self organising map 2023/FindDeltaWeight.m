function deltaWeight = FindDeltaWeight(learningRate, width, winningNeuron,  targetOutput, dataInput, weight)

neighbourhoodFunction =  exp ( (- norm(winningNeuron - targetOutput) ^ 2) / (2 * width ^ 2));
deltaWeight = learningRate * neighbourhoodFunction * (dataInput - weight);

end