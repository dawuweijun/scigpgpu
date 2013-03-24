    // =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2013 - Scilab Enterprises - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

stacksize('max');

function bench()

    lines(0);

    abs_path = get_absolute_file_path("gpuInsert.tst");
    fileName = abs_path + "benchData/insert.h5";

    matSize = 0;
    n       = 200;
    step    = 1250;
    m       = 30;

    computeTime = zeros(n,2);
    dataSize    = zeros(n,1);

    a = rand(2000,2000);
    da = gpuSetData(a);

    for i = 1:n,
        matSize = matSize + step;
        disp(matSize);
        tic();toc();
        b = 1:matSize;
        db = gpuSetData(b);

        dataSize(i,1) = length(b)/1000;

        for j=1:m
            tic(); gpuInsert(da, db, db);    t = toc();
            computeTime(i,2) = computeTime(i,2) + t;
        end

        for j=1:m
            tic();a(b)=b; t = toc();
            computeTime(i,1) = computeTime(i,1) + t;
        end
        db=gpuFree(db);
    end

    computeTime = computeTime / m;
    computeTime = computeTime * 1000;

    da=gpuFree(da);

    saveMyBench(fileName,"Scilab - Insertion a matrix in a part of another matrix","Size of matrix inserted (Thousand of elements)","Computation Time (milliseconds)",..
                dataSize,computeTime,["Intel Xeon CPU E5410 @ 2.33GHz", "Nvidia GTX 560 Ti without transfert"]);

endfunction

bench();
clear bench;
