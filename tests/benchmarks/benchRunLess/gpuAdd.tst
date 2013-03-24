// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    lines(0);

    abs_path = get_absolute_file_path("gpuAdd.tst");
    fileName = abs_path + "benchData/add.h5";

    matSize = 0;
    n       = 40;
    step    = 20;
    m       = 10;

    computeTime = zeros(n,5);
    dataSize    = zeros(n,1);

    for i = 1:n,

        matSize = matSize + step;
        disp(matSize);
        tic();toc();
        b = rand(matSize,matSize);
        a = rand(matSize,matSize);
        dataSize(i,1) = length(a)/1000000;

        for j = 1:m
            tic(); da = gpuSetData(a); db = gpuSetData(b); t = toc();
            da = gpuFree(da);
            db = gpuFree(db);
            computeTime(i,3) = computeTime(i,3) + t;
        end
        da = gpuSetData(a);
        db = gpuSetData(b);

        for j = 1:m
            tic(); dgpu = gpuAdd(da,db); t = toc();
            dgpu = gpuFree(dgpu);
            computeTime(i,4) = computeTime(i,4) + t;
        end
        dgpu = gpuAdd(da,db);

        for j = 1:m
            tic(); gpu = gpuGetData(dgpu); t = toc();
            computeTime(i,5) = computeTime(i,5) + t;
        end

        gpuFree(da);
        gpuFree(db);
        gpuFree(dgpu);

        computeTime(i,2) = computeTime(i,4) + computeTime(i,3) + computeTime(i,5);

        for j = 1:m
            tic();cpu = a + b; t = toc();
            computeTime(i,1) = computeTime(i,1) + t;
        end
    end

    computeTime = computeTime / m;
    computeTime = computeTime * 1000;

    saveMyBench(fileName,"Scilab - Matrix addition","Data Size (Million of elements)","Computation Time (milliseconds)",dataSize,computeTime,["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation","Data transfert GPU to CPU"]);

endfunction

bench();
clear bench;
