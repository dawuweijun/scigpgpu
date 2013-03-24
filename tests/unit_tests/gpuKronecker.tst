// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) Scilab Enterprises - 2013 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

A = rand(3,2);
B = rand(2,4);

//------------ Check the first option------------
cpuRes = A  .*.  B;

d = gpuKronecker(A, B);
h = gpuGetData(d);
gpuFree(d);

assert_checkequal(cpuRes, h);

//------------ Real/Complex------------
ca = rand(3,4) + %i * rand(3,4);
cb = 2 * rand(4,2) + %i * 3 * rand(4,2);
c  = rand(3,4);
d  = 2 * rand(4,2);

gpu = gpuKronecker(2,3);
assert_checkequal(gpuGetData(gpu), (2 .*. 3));
gpuFree(gpu);
gpu = gpuKronecker(c,3);
assert_checkequal(gpuGetData(gpu), (c .*. 3));
gpuFree(gpu);
gpu = gpuKronecker(2,d);
assert_checkequal(gpuGetData(gpu), (2 .*. d));
gpuFree(gpu);
gpu = gpuKronecker(c,d);
assert_checkequal(gpuGetData(gpu), (c .*. d));
gpuFree(gpu);

gpu = gpuKronecker(2*%i,3*%i);
assert_checkequal(gpuGetData(gpu), (2*%i .*. 3*%i));
gpuFree(gpu);
gpu = gpuKronecker(ca,3*%i);
assert_checkequal(gpuGetData(gpu), (ca .*. 3*%i));
gpuFree(gpu);
gpu = gpuKronecker(2*%i,cb);
assert_checkequal(gpuGetData(gpu), (2*%i .*. cb));
gpuFree(gpu);
gpu = gpuKronecker(ca,cb);
assert_checkalmostequal(gpuGetData(gpu), (ca .*. cb));
gpuFree(gpu);

gpu = gpuKronecker(2,3*%i);
assert_checkequal(gpuGetData(gpu), (2 .*. 3*%i));
gpuFree(gpu);
gpu = gpuKronecker(c,3*%i);
assert_checkequal(gpuGetData(gpu), (c .*. 3*%i));
gpuFree(gpu);
gpu = gpuKronecker(2,cb);
assert_checkequal(gpuGetData(gpu), (2 .*. cb));
gpuFree(gpu);
gpu = gpuKronecker(c,cb);
assert_checkequal(gpuGetData(gpu), (c .*. cb));
gpuFree(gpu);

gpu = gpuKronecker(2*%i,3);
assert_checkequal(gpuGetData(gpu), (2*%i .*. 3));
gpuFree(gpu);
gpu = gpuKronecker(ca,3);
assert_checkequal(gpuGetData(gpu), (ca .*. 3));
gpuFree(gpu);
gpu = gpuKronecker(2*%i,d);
assert_checkequal(gpuGetData(gpu), (2*%i .*. d));
gpuFree(gpu);
gpu = gpuKronecker(ca,d);
assert_checkequal(gpuGetData(gpu), (ca .*. d));
gpuFree(gpu);


stacksize('max');
a=1:200000;
b=rand(10,2);
gpu = gpuKronecker(a,b);
assert_checkequal(gpuGetData(gpu), a .*. b);
gpuFree(gpu);

a=1:20000;
b=rand(10,2);
gpu = gpuKronecker(b,a);
assert_checkequal(gpuGetData(gpu), b .*. a);
gpuFree(gpu);

