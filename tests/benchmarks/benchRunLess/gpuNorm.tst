// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    abs_path = get_absolute_file_path("gpuNorm.tst");
    fileName = abs_path + "benchData/norm.h5";

    XSize = [100,200,300,400,500,600,700,800,900,1000];
    YSize = [100,200,300,400,500,600,700,800,900,1000];

    computeTime = zeros(length(XSize),2);
    dataSize    = zeros(length(XSize),1);

    for i = 1:length(XSize),

        a=rand(XSize(i),YSize(i));
        dataSize(i,1) = length(a)/1000000;
        tic();toc();
        tic();gpuNorm(a);    computeTime(i,2) = toc();
        tic();norm(a,'fro'); computeTime(i,1) = toc();

    end

    saveMyBench(fileName,"Scilab - Norm of Matrix","Data Size (Million of elements)","Computation Time (seconds)",dataSize,computeTime,["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti"]);

endfunction

bench();
clear bench;
