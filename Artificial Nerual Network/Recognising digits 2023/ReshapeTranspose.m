function runResultReshapedTransposed = ReshapeTranspose(runResult)

runResultReshaped = reshape(runResult,10,16);
runResultReshapedTransposed = transpose(runResultReshaped);

end