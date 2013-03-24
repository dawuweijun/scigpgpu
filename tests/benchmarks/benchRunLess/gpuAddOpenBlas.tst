// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    abs_path = get_absolute_file_path("gpuAddOpenBlas.tst");
    fileName = abs_path + "benchData/add.h5";
    import_from_hdf5(fileName);

    matSize = xData(1,1);
    n       = length(xData);
    step    = sqrt(xData(n,1)*1000000)/n;

    computeTime = zeros(n,1);

    s = size(legendeString);
    legendeString(s(2)+1) = "Intel Xeon CPU with OpenBlas/GotoBlas";

    for i = 1:n,

        matSize = matSize + step;
        tic();toc();
        b = rand(matSize,matSize);
        a = rand(matSize,matSize);
        cpu = zeros(matSize,matSize);
        dataSize(i,1) = length(a)/1000000;

        tic();cpu = a + b; computeTime(i,1) = toc();

    end

    s = size(yData);
    yData(1:n,s(2)+1) = computeTime(1:n);

    saveMyBench(fileName,titleGraph, xLabel, yLabel,xData,yData,legendeString);

endfunction

bench();
clear bench;
