// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    lines(0);

    abs_path = get_absolute_file_path("gpuMult.tst");
    fileName = abs_path + "benchData/mult.h5";

    matSize = 0;
    n       = 10;
    step    = 100;
    m       = 10;

    computeTime = zeros(n,2);
    dataSize    = zeros(n,1);

    for i = 1:n,

        matSize = matSize + step;
        disp(matSize);
        tic();toc();
        b = rand(matSize,matSize);
        a = rand(matSize,matSize);
        dataSize(i,1) = length(a)/1000000;

        for k=1:m
            tic();cpu = a * b; t = toc();
            computeTime(i,1) = computeTime(i,1) + t;
        end

        for k=1:m
            tic();d = gpuMult(a,b); gpu = gpuGetData(d); gpuFree(d); t = toc();
            computeTime(i,2) = computeTime(i,2) + t;
        end
    end

    computeTime = computeTime / m;

    saveMyBench(fileName,"Scilab - Matrix multiplication","Data Size (Million of elements)","Computation Time (seconds)",dataSize,computeTime,..
                ["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti"]);

endfunction

bench();
clear bench;
