// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function res = compute(XSize, YSize)
        a   = rand(XSize,YSize) + %i*rand(XSize,YSize);
        gpu = zeros(XSize,YSize);
        cpu = zeros(XSize,YSize);
        res(1,6) = length(a)/1000000;
        tic();toc();

        tic();
            cpu   = fft(a);
        res(1,1) = toc();

        tic();
            da    = gpuSetData(a);
        res(1,3) = toc();

        tic();
            dgpu  = gpuFFT(da);
        res(1,4) = toc();

        tic();
            gpu   = gpuGetData(dgpu);
        res(1,5) = toc();

        res(1,2) = res(1,4) + res(1,3) + res(1,5);

        gpuFree(da);
        gpuFree(dgpu);
        clear cpu;
        clear gpu;
endfunction

function bench()

    abs_path = get_absolute_file_path("gpuFFT.tst");
    fileName = abs_path + "benchData/fft.h5";

    XSize = [3, 100:100:1000];
    YSize = [3, 100:100:1000];

    computeTime = zeros(length(XSize),5);
    dataSize    = zeros(length(XSize),1);

    for i = 1:length(XSize), result(i,:) = compute(XSize(i), YSize(i)); end
disp(result)
    computeTime = result(:,1:5);
    dataSize = result(:,6);

    saveMyBench(fileName,"Scilab - FFT Computation","Data Size (Million of elements)","Computation Time (seconds)",dataSize,computeTime,["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation","Data transfert GPU to CPU"]);
endfunction

bench();
clear bench;
