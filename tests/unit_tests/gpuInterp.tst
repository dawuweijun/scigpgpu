// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) DIGITEO - 2012 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

// <-- ENGLISH IMPOSED -->

x = linspace(0,1,11)';
y = cosh(x-0.5);
d = splin(x,y);
xx = linspace(-0.5,1.5,401)';

y0 = interp(xx,x,y,d,"C0");
yy0 = gpuInterp(xx,x,y,d,"C0");
[rows cols] = gpuSize(yy0);
if or([rows cols] <> [401 1]) then pause,end
if or(gpuGetData(yy0) > 2) then pause,end
if or(gpuGetData(yy0) < 0) then pause,end
assert_checkalmostequal(gpuGetData(yy0), y0, %eps, [], "matrix");
gpuFree(yy0);

y1 = interp(xx,x,y,d,"linear");
yy1 = gpuInterp(xx,x,y,d,"linear");
[rows cols] = gpuSize(yy1);
if or([rows cols] <> [401 1]) then pause,end
if or(gpuGetData(yy1) > 2) then pause,end
if or(gpuGetData(yy1) < 0) then pause,end
assert_checkalmostequal(gpuGetData(yy1), y1, %eps, [], "matrix");
gpuFree(yy1);

y2 = interp(xx,x,y,d,"natural");
yy2 = gpuInterp(xx,x,y,d,"natural");
[rows cols] = gpuSize(yy2);
if or([rows cols] <> [401 1]) then pause,end
if or(gpuGetData(yy2) > 2) then pause,end
if or(gpuGetData(yy2) < 0) then pause,end
assert_checkalmostequal(gpuGetData(yy2), y2, %eps, [], "matrix");
gpuFree(yy2);

y3 = interp(xx,x,y,d,"periodic");
yy3 = gpuInterp(xx,x,y,d,"periodic");
[rows cols] = gpuSize(yy3);
if or([rows cols] <> [401 1]) then pause,end
if or(gpuGetData(yy3) > 2) then pause,end
if or(gpuGetData(yy3) < 0) then pause,end
assert_checkalmostequal(gpuGetData(yy3), y3, %eps, [], "matrix");
gpuFree(yy3);
