// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    abs_path = get_absolute_file_path("gpuSumAtlas.tst");
    fileName = abs_path + "benchData/sum.h5";
    import_from_hdf5(fileName);

    n = length(xData);
    computeTime = zeros(n,1);

    s = size(legendeString);
    legendeString(s(2)+1) = "Intel Xeon CPU with Atlas";

    for i = 1:n,
        matrixSize = sqrt(xData(i,1)*1000000);
        a=rand(matrixSize,matrixSize);
        dataSize(i,1) = length(a)/1000000;
        tic();toc();
        tic();sum(a);    computeTime(i,1) = toc();
    end

    s = size(yData);
    yData(1:n,s(2)+1) = computeTime(1:n);

    saveMyBench(fileName,titleGraph, xLabel, yLabel,xData,yData,legendeString);

endfunction

bench();
clear bench;
