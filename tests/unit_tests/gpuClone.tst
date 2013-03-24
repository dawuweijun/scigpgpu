// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) Scilab Enterprises - 2013 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

A=rand(2,4);
dA = gpuSetData(A);
dB = gpuClone(dA);
assert_checkequal(gpuGetData(dB), gpuGetData(dA));

dA=gpuFree(dA);
assert_checkequal(gpuGetData(dB), A);

dB=gpuFree(dB);

// complex
A=rand(2,4) + rand(2,4)*%i;
dA = gpuSetData(A);
dB = gpuClone(dA);
assert_checkequal(gpuGetData(dB), gpuGetData(dA));

dA=gpuFree(dA);
assert_checkequal(gpuGetData(dB), A);

dB=gpuFree(dB);
