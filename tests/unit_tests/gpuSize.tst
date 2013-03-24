// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) DIGITEO - 2011 - Cedric Delamarre
// Copyright (C) Scilab Enterprises - 2013 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

A=rand(7,9);
dA=gpuSetData(A);

[m,n]=gpuSize(dA);
sizes=gpuSize(dA);

assert_checkequal(m, 7);
assert_checkequal(n, 9);
assert_checkequal(sizes, [7 9]);


// overload of size
[m,n]=size(dA);
sizes=size(dA);
r=size(dA, 'r');
c=size(dA, 'c');
s=size(dA, '*');

assert_checkequal(m, 7);
assert_checkequal(n, 9);
assert_checkequal(sizes, [7 9]);
assert_checkequal(r, 7);
assert_checkequal(c, 9);
assert_checkequal(s, 7*9);

dA=gpuFree(dA);
