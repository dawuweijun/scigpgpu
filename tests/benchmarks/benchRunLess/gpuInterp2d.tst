// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function res = compute(XSize, YSize)

        n = XSize;  // a n x n interpolation grid
        x = linspace(0,2*%pi,n); y = x;
        z = cos(x')*cos(y);
        C = splin2d(x, y, z, "periodic");

        // now evaluate on a bigger domain than [0,2pi]x [0,2pi]
        m = YSize; // discretisation parameter of the evaluation grid
        xx = linspace(-0.5*%pi,2.5*%pi,m); yy = xx;
        [XX,YY] = ndgrid(xx,yy);

        res(1, 6) = length(XX) / 1000000;
        tic();toc();

        tic();[a] = interp2d(XX,YY, x, y, C, "C0"); res(1, 1) = toc();

        tic();
            dXX = gpuSetData(XX);
            dYY = gpuSetData(YY);
            dx  = gpuSetData(x);
            dy  = gpuSetData(y);
            dC  = gpuSetData(C);
        res(1, 3) = toc();

        tic();
            [a] = gpuInterp2d(dXX,dYY, dx, dy, dC, "C0");
        res(1, 4) = toc();
        tic();
            da   = gpuGetData(a);
        res(1, 5) = toc();

        res(1, 2) = res(1, 4) + res(1, 3) + res(1, 5);

        gpuFree(dXX);
        gpuFree(dYY);
        gpuFree(dx);
        gpuFree(dy);
        gpuFree(dC);
        gpuFree(a);

endfunction

function res = computeWithGrad(XSize, YSize)

        n = XSize;  // a n x n interpolation grid
        x = linspace(0,2*%pi,n); y = x;
        z = cos(x')*cos(y);
        C = splin2d(x, y, z, "periodic");

        // now evaluate on a bigger domain than [0,2pi]x [0,2pi]
        m = YSize; // discretisation parameter of the evaluation grid
        xx = linspace(-0.5*%pi,2.5*%pi,m); yy = xx;
        [XX,YY] = ndgrid(xx,yy);

        res(1, 6) = length(XX) / 1000000;
        tic();toc();

        tic();[a z e] = interp2d(XX,YY, x, y, C, "C0"); res(1, 1) = toc();

        tic();
            dXX = gpuSetData(XX);
            dYY = gpuSetData(YY);
            dx  = gpuSetData(x);
            dy  = gpuSetData(y);
            dC  = gpuSetData(C);
        res(1, 3) = toc();

        tic();
            [a z e] = gpuInterp2d(dXX,dYY, dx, dy, dC, "C0");
        res(1, 4) = toc();
        tic();
            da   = gpuGetData(a);
            dz   = gpuGetData(z);
            de   = gpuGetData(e);
        res(1, 5) = toc();

        res(1, 2) = res(1, 4) + res(1, 3) + res(1, 5);

        gpuFree(dXX);
        gpuFree(dYY);
        gpuFree(dx);
        gpuFree(dy);
        gpuFree(dC);
        gpuFree(a);
        gpuFree(z);
        gpuFree(e);

endfunction

function res = computeWithGradAndHes(XSize, YSize)

        n = XSize;  // a n x n interpolation grid
        x = linspace(0,2*%pi,n); y = x;
        z = cos(x')*cos(y);
        C = splin2d(x, y, z, "periodic");

        // now evaluate on a bigger domain than [0,2pi]x [0,2pi]
        m = YSize; // discretisation parameter of the evaluation grid
        xx = linspace(-0.5*%pi,2.5*%pi,m); yy = xx;
        [XX,YY] = ndgrid(xx,yy);

        res(1, 6) = length(XX) / 1000000;
        tic();toc();

        tic();[a z e r t u] = interp2d(XX,YY, x, y, C, "C0"); res(1, 1) = toc();

        tic();
            dXX = gpuSetData(XX);
            dYY = gpuSetData(YY);
            dx  = gpuSetData(x);
            dy  = gpuSetData(y);
            dC  = gpuSetData(C);
        res(1, 3) = toc();

        tic();
            [a z e r t u] = gpuInterp2d(dXX,dYY, dx, dy, dC, "C0");
        res(1, 4) = toc();
        tic();
            da   = gpuGetData(a);
            dz   = gpuGetData(z);
            de   = gpuGetData(e);
            dr   = gpuGetData(r);
            dt   = gpuGetData(t);
            du   = gpuGetData(u);
        res(1, 5) = toc();

        res(1, 2) = res(1, 4) + res(1, 3) + res(1, 5);

        gpuFree(dXX);
        gpuFree(dYY);
        gpuFree(dx);
        gpuFree(dy);
        gpuFree(dC);
        gpuFree(a);
        gpuFree(z);
        gpuFree(e);
        gpuFree(r);
        gpuFree(t);
        gpuFree(u);

endfunction

function bench()

    abs_path = get_absolute_file_path("gpuInterp2d.tst");

    XDim = [7, 14, 28, 70, 140, 280];
    YDim = [80, 160, 320, 800, 1600, 3200];

    computeTime = zeros(length(XDim),5);
    dataSize    = zeros(length(XDim),1);

    for i = 1:length(XDim), result(i,:) = compute(XDim(i), YDim(i)), end;

    computeTime = result(:,1:5);
    dataSize = result(:,6);

    fileName = abs_path + "benchData/interp2d.h5";
    saveMyBench(fileName,"Scilab - 2D Interpolation with C0 outmode.","Data Size (Million of elements)","Computation Time (seconds)",dataSize,computeTime,["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation","Data transfert GPU to CPU"]);

    for i = 1:length(XDim), result(i,:) = computeWithGrad(XDim(i), YDim(i)), end;

    computeTime = result(:,1:5);
    dataSize = result(:,6);

    fileName = abs_path + "benchData/interp2dGrad.h5";
    saveMyBench(fileName,["Scilab - 2D Interpolation with C0 outmode and compute gradient."],"Data Size (Million of elements)","Computation Time (seconds)",dataSize,computeTime,["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation","Data transfert GPU to CPU"]);

    for i = 1:length(XDim), result(i,:) = computeWithGradAndHes(XDim(i), YDim(i)), end;

    computeTime = result(:,1:5);
    dataSize = result(:,6);

    fileName = abs_path + "benchData/interp2dGradHes.h5";
    saveMyBench(fileName,["Scilab - 2D Interpolation with C0 outmode and compute gradient and Hessean."],"Data Size (Million of elements)","Computation Time (seconds)",dataSize,computeTime,["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation","Data transfert GPU to CPU"]);

endfunction

bench();
clear bench;

