// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2012 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function res = computeBis(XSize)

    x = linspace(-XSize,XSize,2*XSize)';
    y = sinc(x);

    df = splin(x,y, "fast");
    xx = linspace(-XSize,XSize,100*XSize)';

    res(1, 6) = length(xx) / 1000000;

    tic();toc();

    tic();[yyf] = interp(xx, x, y, df); res(1, 1) = toc()

    tic();
        dxx = gpuSetData(xx);
        dx = gpuSetData(x);
        dy = gpuSetData(y);
        ddf = gpuSetData(df);
    res(1, 3) = toc();

    tic();
        [d_yyf] = gpuInterp(dxx, dx, dy, ddf);
    res(1, 4) = toc();

    tic();
        dyyf = gpuGetData(d_yyf);
    res(1, 5) = toc();

    res(1, 2) = res(1, 3) + res(1, 4) + res(1, 5);

    gpuFree(d_yyf);
    gpuFree(dxx);
    gpuFree(dx);
    gpuFree(dy);
    gpuFree(ddf);
endfunction

function res = compute(XSize)

    x = linspace(-XSize,XSize,2*XSize)';
    y = sinc(x);

    df = splin(x,y, "fast");
    xx = linspace(-XSize,XSize,100*XSize)';

    res(1, 6) = length(xx) / 1000000;

    tic();toc();

    tic();[yyf, yy1f, yy2f] = interp(xx, x, y, df); res(1, 1) = toc()

    tic();
        dxx = gpuSetData(xx);
        dx = gpuSetData(x);
        dy = gpuSetData(y);
        ddf = gpuSetData(df);
    res(1, 3) = toc();

    tic();
        [d_yyf, d_yy1f, d_yy2f] = gpuInterp(dxx, dx, dy, ddf);
    res(1, 4) = toc();

    tic();
        dyyf = gpuGetData(d_yyf);
        dyy1f = gpuGetData(d_yy1f);
        dyy2f = gpuGetData(d_yy2f);
    res(1, 5) = toc();

    res(1, 2) = res(1, 3) + res(1, 4) + res(1, 5);

    gpuFree(d_yyf);
    gpuFree(d_yy1f);
    gpuFree(d_yy2f);
    gpuFree(dxx);
    gpuFree(dx);
    gpuFree(dy);
    gpuFree(ddf);
endfunction

function bench()

    abs_path = get_absolute_file_path("gpuInterp.tst");

    XDim = [8, 16, 80, 160, 800, 1600, 8000];

    computeTime = zeros(length(XDim),5);
    dataSize    = zeros(length(XDim),1);

    for i = 1:length(XDim), result(i,:) = compute(XDim(i)), end;

    computeTime = result(:,1:5);
    dataSize = result(:,6);
disp(result)
    fileName = abs_path + "benchData/interp.h5";
    computeTime = computeTime * 1000;
    saveMyBench(fileName,"Scilab - Interpolation with C0 outmode and three outputs arguments.","Data Size (Million of elements)","Computation Time (milliseconds)",dataSize,computeTime,..
                ["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation","Data transfert GPU to CPU"]);

    for i = 1:length(XDim), result(i,:) = computeBis(XDim(i)), end;

    computeTime = result(:,1:5);
    dataSize = result(:,6);
    computeTime = computeTime * 1000;
disp(result)
    fileName = abs_path + "benchData/interpOneReturn.h5";
    saveMyBench(fileName,"Scilab - Interpolation with C0 outmode and one output argument.","Data Size (Million of elements)","Computation Time (milliseconds)",dataSize,computeTime,..
                ["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti", "Data transfert CPU to GPU", "GPU Computation","Data transfert GPU to CPU"]);

endfunction

bench();
clear bench;

