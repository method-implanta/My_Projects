function runResult = RunHopfieldNetwork(networkWeight, networkState, x)

for i = 1:160
    networkState(i) = sign(networkWeight(i,:) * transpose(networkState));
    if networkState(i) == 0
        networkState(i) = 1;
    end
end
while isequal(networkState, x)
    for i = 1:160
        networkState(i) = sign(networkWeight(i,:) * transpose(networkState));
        if networkState(i) == 0
            networkState(i) = 1;
        end
    end
end

runResult = networkState;

end