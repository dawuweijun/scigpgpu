// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    abs_path = get_absolute_file_path("gpuTransposeTwice.tst");
    fileName = abs_path + "benchData/transposeTwice.h5";

    XSize = [1000, 2000, 3000, 4000, 5000, 6000];
    YSize = [1000, 2000, 3000, 4000, 5000, 6000];

    computeTime = zeros(length(XSize),5);
    dataSize    = zeros(length(XSize),1);

    for i = 1:length(XSize),

        a   = rand(XSize(i),YSize(i));
        b   = zeros(XSize(i),YSize(i));
        cpu = zeros(XSize(i),YSize(i));
        dataSize(i,1) = length(a)/1000000;
        tic();toc();

        tic();
            b=a';
            b';
        computeTime(i,1) = toc();

        tic(); da = gpuSetData(a); computeTime(i,3) = toc();

        tic();
            gpu     = gpuTranspose(da);
            dres    = gpuTranspose(gpu);
        computeTime(i,4) = toc();
        gpuFree(da);
        gpuFree(gpu);

        tic();cpu   = gpuGetData(dres);         computeTime(i,5) = toc();
        gpuFree(dres);
        computeTime(i,2) = computeTime(i,3) + computeTime(i,4) + computeTime(i,5);

    end

    saveMyBench(fileName,"Scilab - Transpose twice matrix","Data Size (Million of elements)","Computation Time (seconds)",dataSize,computeTime,["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation","Data transfert GPU to CPU"]);

endfunction

bench();
clear bench;
