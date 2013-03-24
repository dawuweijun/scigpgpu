// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

//==============================================================================
// Benchmark for multiple product on GPU
//==============================================================================

// <-- BENCH NB RUN : 10 -->

stacksize('max');

A=rand(1000,1000);
B=rand(1000,1000);

dA=gpuSetData(A);
dB=gpuSetData(B);

n = 100

// <-- BENCH START -->

GPU=gpuMult(A,B);

for i = 0:n,
    GPU1=gpuMult(GPU,dB);GPU=gpuFree(GPU);
    GPU=gpuMult(dA,GPU1);GPU1=gpuFree(GPU1);
end;

GPU1=gpuMult(GPU,dB);GPU=gpuFree(GPU);
GPU=gpuMult(dA,GPU1);GPU1=gpuFree(GPU1);
CPU=getGetData(GPU);GPU=gpuFree(GPU);

// <-- BENCH END -->

dA=gpuFree(dA);
dB=gpuFree(dB);
