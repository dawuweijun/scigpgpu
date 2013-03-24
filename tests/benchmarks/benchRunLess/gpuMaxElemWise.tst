// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2013 - Scialb Enterprises - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    lines(0);

    abs_path = get_absolute_file_path("gpuMaxElemWise.tst");
    fileName = abs_path + "benchData/maxelemwise.h5";

    matSize = 0;
    n       = 50;
    step    = 10;
    m       = 10;

    computeTime = zeros(n,5);
    dataSize    = zeros(n,1);

    for i = 1:n,

        matSize = matSize + step;
        disp(matSize);
        tic();toc();
        a = rand(matSize,matSize);
        b = rand(matSize,matSize);

        dataSize(i,1) = length(a)/1000;

        for j=1:m
            tic(); da = gpuSetData(a); t = toc();
            tic(); db = gpuSetData(b); t = toc();
            da=gpuFree(da);
            da=gpuFree(db);
            computeTime(i,3) = computeTime(i,3) + t;
        end
        da = gpuSetData(a);

        for j=1:m
            tic(); dgpu = gpuMax(da,db); t = toc();
            dgpu=gpuFree(dgpu);
            computeTime(i,4) = computeTime(i,4) + t;
        end
        dgpu = gpuMax(da,db);

        for j=1:m
            tic(); gpu  = gpuGetData(dgpu); t = toc();
            computeTime(i,5) = computeTime(i,5) + t;
        end

        gpuFree(da);
        gpuFree(dgpu);

        computeTime(i,2) = computeTime(i,4) + computeTime(i,3) + computeTime(i,5);

        for j=1:m
            tic();cpu = max(a,b); t = toc();
            computeTime(i,1) = computeTime(i,1) + t;
        end
    end

    computeTime = computeTime / m;
    computeTime = computeTime * 1000;

    saveMyBench(fileName,"Scilab - The maximum between two matrix","Data Size (Thousand of elements)","Computation Time (milliseconds)",dataSize,computeTime,..
                ["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation" "Data transfert GPU to CPU"]);

endfunction

bench();
clear bench;
