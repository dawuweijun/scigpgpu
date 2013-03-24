// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) DIGITEO - 2011 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

// <-- ENGLISH IMPOSED -->

A=rand(2,4);
B=rand(2,4);

cpuRes = A + B;
gpuRes = gpuAdd(A, B);

assert_checkalmostequal(cpuRes, gpuGetData(gpuRes), %eps);

// ------------- Real/Complex -------------

ca = rand(4,6) + %i * rand(4,6);
cb = 2 * rand(4,6) + %i * 3 * rand(4,6);
c  = rand(4,6);
d  = 2 * rand(4,6);

gpu = gpuAdd(2,3);
assert_checkalmostequal(gpuGetData(gpu), 2 + 3, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(c,3);
assert_checkalmostequal(gpuGetData(gpu), c+3, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(2,d);
assert_checkalmostequal(gpuGetData(gpu), 2+d, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(c,d);
assert_checkalmostequal(gpuGetData(gpu), c+d, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(2*%i,3*%i);
assert_checkalmostequal(gpuGetData(gpu), 2*%i+3*%i, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(ca,3*%i);
assert_checkalmostequal(gpuGetData(gpu), ca+3*%i, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(2*%i,cb);
assert_checkalmostequal(gpuGetData(gpu), 2*%i+cb, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(ca,cb);
assert_checkalmostequal(gpuGetData(gpu), ca+cb, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(2,3*%i);
assert_checkalmostequal(gpuGetData(gpu), 2+3*%i, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(c,3*%i);
assert_checkalmostequal(gpuGetData(gpu), c+3*%i, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(2,cb);
assert_checkalmostequal(gpuGetData(gpu), 2+cb, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(c,cb);
assert_checkalmostequal(gpuGetData(gpu), c+cb, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(2*%i,3);
assert_checkalmostequal(gpuGetData(gpu), 2*%i+3, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(ca,3);
assert_checkalmostequal(gpuGetData(gpu), ca+3, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(2*%i,d);
assert_checkalmostequal(gpuGetData(gpu), 2*%i+d, %eps);
gpu = gpuFree(gpu);

gpu = gpuAdd(ca,d);
assert_checkalmostequal(gpuGetData(gpu), ca+d, %eps);
gpu = gpuFree(gpu);


// ------------- some tests -------------
stacksize(50000000);
a = rand(256,256);
ierr = execstr("d = gpuAdd(a,a); gpuFree(d);","errcatch");
if ierr == 999 then pause,end;

a = rand(3000,3000);
ierr = execstr("d = gpuAdd(a,a); gpuFree(d);","errcatch");
if ierr == 999 then pause,end;

