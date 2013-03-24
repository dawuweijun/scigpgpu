    // =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2013 - Scilab Enterprises - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    lines(0);

    abs_path = get_absolute_file_path("gpuExtract.tst");
    fileName = abs_path + "benchData/extract.h5";

    matSize = 0;
    n       = 100;
    step    = 10000;
    m       = 10;

    computeTime = zeros(n,5);
    dataSize    = zeros(n,1);

    a = rand(3000,3000);
    da = gpuSetData(a);

    for i = 1:n,

        matSize = matSize + step;
        disp(matSize);
        tic();toc();
        b = 1:matSize;

        dataSize(i,1) = length(b)/1000;

        for j=1:m
            tic(); db = gpuSetData(b); t = toc();
            db=gpuFree(db);
            computeTime(i,3) = computeTime(i,3) + t;
        end
        db = gpuSetData(b);

        for j=1:m
            tic(); dgpu = gpuExtract(da,db);    t = toc();
            dgpu=gpuFree(dgpu);
            computeTime(i,4) = computeTime(i,4) + t;
        end
        dgpu = gpuExtract(da,db);

        for j=1:m
            tic(); gpu  = gpuGetData(dgpu); t = toc();
            computeTime(i,5) = computeTime(i,5) + t;
        end

        gpuFree(db);
        gpuFree(dgpu);

        computeTime(i,2) = computeTime(i,4) + computeTime(i,3) + computeTime(i,5);

        for j=1:m
            tic();cpu = a(b); t = toc();
            computeTime(i,1) = computeTime(i,1) + t;
        end
    end

    computeTime = computeTime / m;
    computeTime = computeTime * 1000;

    gpuFree(da);

    saveMyBench(fileName,"Scilab - Extraction of a part of matrix","Size of data extracted(Thousand of elements)","Computation Time (milliseconds)",..
                dataSize,computeTime,["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation","Data transfert GPU to CPU"]);

endfunction

bench();
clear bench;
