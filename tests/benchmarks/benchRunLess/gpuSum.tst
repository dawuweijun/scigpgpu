// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    abs_path = get_absolute_file_path("gpuSum.tst");
    fileName = abs_path + "benchData/sum.h5";

    XSize = [1000, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000];
    YSize = [1000, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000];

    computeTime = zeros(length(XSize),4);
    dataSize    = zeros(length(XSize),1);

    for i = 1:length(XSize),

        a=rand(XSize(i),YSize(i));
        dataSize(i,1) = length(a)/1000000;
        tic();toc();

        tic();sum(a);    computeTime(i,1) = toc();

        tic();da = gpuSetData(a);   computeTime(i,3) = toc();
        tic();gpuSum(da);           computeTime(i,4) = toc();

        computeTime(i,2) = computeTime(i,4) + computeTime(i,3);
        gpuFree(da);
    end

    saveMyBench(fileName,"Scilab - Sum of Matrix","Data Size (Million of elements)","Computation Time (seconds)",dataSize,computeTime,..
                ["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation"]);

endfunction

bench();
clear bench;
