// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    lines(0);

    abs_path = get_absolute_file_path("gpuClone.tst");
    fileName = abs_path + "benchData/clone.h5";

    matSize = 0;
    n       = 50;
    step    = 20;
    m       = 10;

    computeTime = zeros(n,2);
    dataSize    = zeros(n,1);

    for i = 1:n,
        matSize = matSize + step;
        disp(matSize);
        a = rand(matSize,matSize);
        dataSize(i,1) = length(a)/1000000;
        da = gpuSetData(a);

        for j = 1:m
            tic();cpu = a; t = toc();
            computeTime(i,1) = computeTime(i,1) + t;

            tic(); dgpu = gpuClone(da); t = toc();
            gpuFree(dgpu);
            computeTime(i,2) = computeTime(i,2) + t;
        end

        gpuFree(da);
    end

    computeTime = computeTime / m;
    computeTime = computeTime * 1000;

    saveMyBench(fileName,"Scilab - Matrix clone","Data Size (Million of elements)","Computation Time (milliseconds)",dataSize,computeTime,..
                ["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti"]);

endfunction

bench();
clear bench;
