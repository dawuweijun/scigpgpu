// =============================================================================
// Scilab ( http://www.scilab.org/ ), This file is part of Scilab
// Copyright (C) Scilab Enterprises - 2013 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

// input vector
a  = rand(5,31);
da = gpuSetData(a);

// matrix input
b = ones(a);

db = gpuOnes(a);
assert_checkequal(gpuGetData(db), b);
assert_checkequal(size(db), size(b));
gpuFree(db);

db = gpuOnes(da);
assert_checkequal(gpuGetData(db), b);
assert_checkequal(size(db), size(b));
gpuFree(db);

// size input
b = ones(12, 46);
db = gpuOnes(12, 46);
assert_checkequal(gpuGetData(db), b);
assert_checkequal(size(db), size(b));
gpuFree(db);

