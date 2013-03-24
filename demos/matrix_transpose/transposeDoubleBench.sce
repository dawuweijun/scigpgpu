// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

lines(0);
stacksize('max');
function tansposeDoubleBench()

    abs_path=get_absolute_file_path("transposeDoubleBench.sce");

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
        cpu = zeros(XSize(i),YSize(i));
        h_c = zeros(XSize(i),YSize(i));
        dataSize(i,1) = length(h_a)/1000000;
        tic();toc();

        tic();
            t1 = h_a';
            t2 = t1';
        toc();

        tic();
            t1 = h_a';
            t2 = t1';
        computeTime(i,1) = toc();

        tic();
            t1 = gpuTranspose(h_a);
            t2 = gpuTranspose(t1);
            cpu = gpuGetData(t2)
            gpuFree(t1);
            gpuFree(t2);
        computeTime(i,2) = toc();

        tic();
            d_a=gpuSetData(h_a);
            d_c=gpuAlloc(YSize(i),XSize(i));

            lst=list(d_c, d_a, int32(YSize(i)), int32(XSize(i)));
            gpuApplyFunction(fonc, lst, H_W_Block, H_W_Block, H_W_Grid(i) ,H_W_Grid(i)); // fonc,lst,block_h,block_w,grid_h,grid_w

            lst=list(d_a, d_c, int32(YSize(i)), int32(XSize(i)));
            gpuApplyFunction(fonc, lst, H_W_Block, H_W_Block, H_W_Grid(i) ,H_W_Grid(i)); // fonc,lst,block_h,block_w,grid_h,grid_w

            h_c=gpuGetData(d_a);

            gpuFree(d_c);
            gpuFree(d_a);
        computeTime(i,3) = toc();

        if ~and(cpu == h_a) then pause, end
        if ~and(h_c == h_a) then pause, end
    end

    saveMyBench(abs_path+"result/transposeTwoTimes.h5","Transpose two times the same matrix","Size of datas (Millon of elements)","Compute Time (second)",dataSize,computeTime,["CPU", "GPU", "GPU Kernel"]);

    if or(getscilabmode() == ["NW";"STD"]) then
        showMyBench(abs_path+"result/transposeTwoTimes.h5");
    end

endfunction

tansposeDoubleBench();
clear tansposeDoubleBench;
