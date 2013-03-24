// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2013 - Scilab Enterprises - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    lines(0);

    abs_path = get_absolute_file_path("gpuKronecker.tst");
    fileName = abs_path + "benchData/kronecker.h5";

    matSize = 0;
    n       = 15;
    step    = 4;
    m       = 10;

    computeTime = zeros(n,5);
    dataSize    = zeros(n,1);

    for i = 1:n,

        matSize = matSize + step;
        disp(matSize);
        tic();toc();
        b = rand(matSize,matSize);
        a = rand(matSize,matSize);
        dataSize(i,1) = length(a);

        for k=1:m
            tic();cpu = a .*. b; t = toc();
            computeTime(i,1) = computeTime(i,1) + t;
        end

        gpu = zeros(matSize*matSize,matSize*matSize);
        for k=1:m
            tic(); da=gpuSetData(a); db=gpuSetData(b); t = toc();
            computeTime(i,3) = computeTime(i,3) + t;
            tic();d = gpuKronecker(da,db); t = toc();
            computeTime(i,4) = computeTime(i,4) + t;
            tic(); gpu = gpuGetData(d); t = toc();
            computeTime(i,5) = computeTime(i,5) + t;

            gpuFree(d);
            gpuFree(da);
            gpuFree(db);
        end
        computeTime(i,2) = computeTime(i,3) + computeTime(i,4) + computeTime(i,5);
    end

    computeTime = computeTime / m;
    computeTime = computeTime * 1000;

    saveMyBench(fileName,"Scilab - Matrix kronecker multiplication - gpuKronecker(A,B)","Number of elements in A (same as B)","Computation Time (milliseconds)",dataSize,computeTime,..
                ["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Transfert Host to Device", "Device computation", "Transfert Device to Host"]);

endfunction

bench();
clear bench;
