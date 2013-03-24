// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2013 - Scialb Enterprises - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    lines(0);

    abs_path = get_absolute_file_path("gpuMax.tst");
    fileName = abs_path + "benchData/max.h5";

    matSize = 0;
    n       = 100;
    step    = 10;
    m       = 10;

    computeTime = zeros(n,4);
    dataSize    = zeros(n,1);

    for i = 1:n,

        matSize = matSize + step;
        disp(matSize);
        tic();toc();
        a = rand(matSize,matSize);

        dataSize(i,1) = length(a)/1000;

        for j=1:m
            tic(); da = gpuSetData(a); t = toc();
            da=gpuFree(da);
            computeTime(i,3) = computeTime(i,3) + t;
        end
        da = gpuSetData(a);

        for j=1:m
            tic(); dgpu = gpuMax(da); t = toc();
            computeTime(i,4) = computeTime(i,4) + t;
        end

        gpuFree(da);

        computeTime(i,2) = computeTime(i,4) + computeTime(i,3);

        for j=1:m
            tic();cpu = max(a); t = toc();
            computeTime(i,1) = computeTime(i,1) + t;
        end
    end

    computeTime = computeTime / m;
    computeTime = computeTime * 1000;

    saveMyBench(fileName,"Scilab - The maximum of a matrix","Data Size (Thousand of elements)","Computation Time (milliseconds)",dataSize,computeTime,..
                ["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation"]);

endfunction

bench();
clear bench;
