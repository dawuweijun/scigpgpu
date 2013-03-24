// =============================================================================
// Scilab ( http://www.scilab.org/ ), This file is part of Scilab
// Copyright (C) DIGITEO, 2011, Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

// input vector
a  = 1:8;
da = gpuSetData(a);

db = gpuMatrix(a,2,4);
dc = gpuMatrix(da,2,4);
dd = gpuMatrix(da,2,-1);

[r,c]=gpuSize(db);
assert_checkequal(r,2);
assert_checkequal(c,4);
[r,c]=gpuSize(dc);
assert_checkequal(r,2);
assert_checkequal(c,4);
[r,c]=gpuSize(dd);
assert_checkequal(r,2);
assert_checkequal(c,4);

da=gpuFree(da);
db=gpuFree(db);
dc=gpuFree(dc);
dd=gpuFree(dd);

// input matrix
a  = matrix(1:8,4,2);
da = gpuSetData(a);

db = gpuMatrix(a,2,4);
dc = gpuMatrix(da,2,4);
dd = gpuMatrix(da,-1,4);

[r,c]=gpuSize(db);
assert_checkequal(r,2);
assert_checkequal(c,4);
[r,c]=gpuSize(dc);
assert_checkequal(r,2);
assert_checkequal(c,4);
[r,c]=gpuSize(dd);
assert_checkequal(r,2);
assert_checkequal(c,4);

db=gpuFree(db);
dc=gpuFree(dc);
dc=gpuFree(dd);

// vector of sizes
db = gpuMatrix(a,[2,4]);
dc = gpuMatrix(da,[2,4]);

[r,c]=gpuSize(db);
assert_checkequal(r,2);
assert_checkequal(c,4);
[r,c]=gpuSize(dc);
assert_checkequal(r,2);
assert_checkequal(c,4);

da=gpuFree(da);
db=gpuFree(db);
dc=gpuFree(dc);

