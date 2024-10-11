function PrintResult(runResultReshapedTransposed)

formatSpecFirst = '[[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d], ';
formatSpecMiddle = '[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d], ';
formatSpecLast = '[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d]]';
for j = 1:16
    if j == 1
        fprintf(formatSpecFirst, runResultReshapedTransposed(j,:));
    elseif j == 16
            fprintf(formatSpecLast, runResultReshapedTransposed(j,:));
    else
        fprintf(formatSpecMiddle, runResultReshapedTransposed(j,:));
    end
end

end