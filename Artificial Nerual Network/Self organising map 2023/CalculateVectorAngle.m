function vectorAngle = CalculateVectorAngle(weight, dataInput)

vectorAngle = acos(dot(weight, dataInput)/(norm(weight) * norm(dataInput)));

end