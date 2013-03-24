// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2013 - Scilab Enterprises - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    lines(0);

    abs_path = get_absolute_file_path("gpuSplin2d.tst");
    fileName = abs_path + "benchData/splin2d.h5";

    n       = 500;
    m       = 10;

    computeTime = zeros(size(2:10:n,'*'),5);
    dataSize    = zeros(size(2:10:n,'*'),1);
    iter = 0;
    for i = 2:10:n,
        iter = iter + 1;
        x = linspace(0,2*%pi,i); y = x;
        z = cos(x')*cos(y);
        disp(i);
        dataSize(iter,1) = i;
        t=0;
        for k=1:m
            tic();C = splin2d(x, y, z, "natural");t=toc();
            computeTime(iter,1) = computeTime(iter,1) + t;
        end



        for k=1:m
            tic();
            dx = gpuSetData(x);
            dy = gpuSetData(y);
            dz = gpuSetData(z);
            t = toc();
            computeTime(iter,3) = computeTime(iter,3) + t;
            tic();dC = gpuSplin2d(dx, dy, dz, "natural");t = toc();
            computeTime(iter,4) = computeTime(iter,4) + t;
            tic();cc=gpuGetData(dC);t = toc();
            computeTime(iter,5) = computeTime(iter,5) + t;
            gpuFree(dC);
            gpuFree(dx);
            gpuFree(dy);
            gpuFree(dz);
        end
        computeTime(iter,2) = computeTime(iter,3) + computeTime(iter,4) + computeTime(iter,5);
    end

    computeTime = computeTime / m;

    saveMyBench(fileName,"Scilab - Splin2d(X,Y,Z,""natural"")","Size of vector X and Y","Computation Time (seconds)",dataSize,computeTime,..
                ["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Transfert host to device", "Device Computation", "Transfert device to host"]);

endfunction

bench();
clear bench;
