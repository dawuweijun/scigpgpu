// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) Scilab Enterprises - 2013 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

// <-- ENGLISH IMPOSED -->

A = rand(2,4);
B = rand(2,4);
dA = gpuSetData(A);
dB = gpuSetData(B);

cpuRes = complex(A, B);
gpuRes = gpuComplex(A, B);
assert_checkequal(cpuRes, gpuGetData(gpuRes));
gpuFree(gpuRes);

cpuRes = complex(A, B);
gpuRes = gpuComplex(dA, dB);
assert_checkequal(cpuRes, gpuGetData(gpuRes));
gpuFree(gpuRes);

cpuRes = complex(A, 5);
gpuRes = gpuComplex(dA, 5);
assert_checkequal(cpuRes, gpuGetData(gpuRes));
gpuFree(gpuRes);

cpuRes = complex(4, B);
gpuRes = gpuComplex(4, B);
assert_checkequal(cpuRes, gpuGetData(gpuRes));
gpuFree(gpuRes);

gpuFree(dA);
gpuFree(dB);
