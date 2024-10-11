function networkWeight = HebbsRule(xCombination)

networkWeight = zeros(160);
for i = 1:5
    networkWeight = networkWeight + (1/160) * transpose(xCombination(i,:)) * xCombination(i,:);
end
for j = 1:160
    for k = 1:160
        if j == k
            networkWeight(j,k) = 0;
        end
    end
end


end