// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

lines(0);
stacksize('max');
function tansposeBench()

    abs_path=get_absolute_file_path("transposeBench.sce");

    bin=gpuBuild(abs_path + "transpose_kernel");
    fonc=gpuLoadFunction(bin,"transpose_naive");

    XSize = [1000, 1500, 2000, 2500];
    YSize = [1000, 1500, 2000, 2500];

    H_W_Block = 16;
    H_W_Grid  = [63, 94, 125, 157];

    computeTime = zeros(length(XSize),3);
    dataSize    = zeros(length(XSize),1);

    for i = 1:length(XSize),

        h_a = rand(XSize(i),YSize(i));
        b   = zeros(XSize(i),YSize(i));
        cpu = zeros(XSize(i),YSize(i));
        h_c = zeros(XSize(i),YSize(i));
        dataSize(i,1) = length(h_a)/1000000;
        tic();toc();

        tic();t1=gpuTranspose(h_a);cpu=gpuGetData(t1);      computeTime(i,2) = toc();
        tic();b=h_a';                     computeTime(i,1) = toc();

        tic();
            d_a=gpuSetData(h_a);
            d_c=gpuAlloc(YSize(i),XSize(i));

            lst=list(d_c, d_a, int32(YSize(i)), int32(XSize(i)));
            gpuApplyFunction(fonc, lst, H_W_Block, H_W_Block, H_W_Grid(i) ,H_W_Grid(i)); // fonc,lst,block_h,block_w,grid_h,grid_w

            h_c=gpuGetData(d_c);

            gpuFree(d_c);
            gpuFree(d_a);
        computeTime(i,3) = toc();

        if ~and(cpu == b) then pause, end
        if ~and(h_c == b) then pause, end
    end

    // save results
    saveMyBench(abs_path+"result/oneTranspose.h5","Transpose a matrix","Size of datas (Millon of elements)","Compute Time (second)",dataSize,computeTime,["CPU", "GPU", "GPU Kernel"]);

    // display results
    if or(getscilabmode() == ["NW";"STD"]) then
        showMyBench(abs_path+"result/oneTranspose.h5");
    end
endfunction

tansposeBench();
clear tansposeBench;
