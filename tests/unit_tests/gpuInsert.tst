// =============================================================================
// Scilab ( http://www.scilab.org/ ), This file is part of Scilab
// Copyright (C) Scilab Enterprises, 2013, Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

a  = matrix(1:100, 10, 10);
da = gpuSetData(a);

// insert a scalar
gpuInsert(da, 3000, 3, 3);
a(3, 3) = 3000;
assert_checkequal(gpuGetData(da), a);

da(4,4) = 4000;
a(4,4) = 4000;
assert_checkequal(gpuGetData(da), a);

// insert a matrix into a matrix
gpuInsert(da,-10, [1 6 23 64]);
a([1 6 23 64]) = -10;
assert_checkequal(gpuGetData(da), a);

da([1 6 23 64]) = -11;
a([1 6 23 64]) = -11;
assert_checkequal(gpuGetData(da), a);

gpuInsert(da, (1:11:100) * -1, 1:11:100);
a(1:11:100) = (1:11:100) * -1;
assert_checkequal(gpuGetData(da), a);

da([1:11:100]) = (1:11:100) * -2;
a([1:11:100]) = (1:11:100) * -2;
assert_checkequal(gpuGetData(da), a);

db = gpuSetData([1:11:100]);
dc = gpuSetData((1:11:100) * -3);
da(db) = dc;
a([1:11:100]) = (1:11:100) * -3;
assert_checkequal(gpuGetData(da), a);

gpuFree(db);
gpuFree(dc);

// insertion only allowed with overload
da(:) = 0;
a(:) = 0;
assert_checkequal(gpuGetData(da), a);

da(2,:) = 2;
a(2,:) = 2;
assert_checkequal(gpuGetData(da), a);
da(:,4) = %i;
a(:,4) = %i;
assert_checkequal(gpuGetData(da), a);

da([1 2; 4 5], [1 2 3]) = 36 + 6*%i;
a([1 2; 4 5], [1 2 3]) = 36 + 6*%i;
assert_checkequal(gpuGetData(da), a);

da=gpuFree(da);
