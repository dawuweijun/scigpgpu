// =============================================================================
// Scilab ( http://www.scilab.org/ ), This file is part of Scilab
// Copyright (C) Scilab Enterprises - 2013 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

x  = 1:9;
y  = 1:5;

[dX dY] = gpuNdgrid(x,y);
[X Y] = ndgrid(x,y);

assert_checkequal(gpuGetData(dX), X);
assert_checkequal(gpuGetData(dY), Y);

gpuFree(dX);
gpuFree(dY);
