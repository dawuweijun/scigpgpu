// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) DIGITEO - 2012 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

// <-- ENGLISH IMPOSED -->

n = 7;  // a n x n interpolation grid
x = linspace(0,2*%pi,n); y = x;
z = cos(x')*cos(y);
C = splin2d(x, y, z, "periodic");

// now evaluate on a bigger domain than [0,2pi]x [0,2pi]
m = 80; // discretisation parameter of the evaluation grid
xx = linspace(-0.5*%pi,2.5*%pi,m); yy = xx;
[XX,YY] = ndgrid(xx,yy);

outmode = "C0";
gpuMem = gpuDeviceMemInfo();
[zz1 dzpdx dzpdy d2Zdxx d2Zdxy d2Zdyy]= interp2d(XX,YY, x, y, C, outmode);
[gzz1 gdzpdx gdzpdy gd2Zdxx gd2Zdxy gd2Zdyy] = gpuInterp2d(XX,YY, x, y, C, outmode);
dzz1 = gpuGetData(gzz1); gpuFree(gzz1);
ddzpdx = gpuGetData(gdzpdx); gpuFree(gdzpdx);
ddzpdy = gpuGetData(gdzpdy); gpuFree(gdzpdy);
d2Zd2x = gpuGetData(gd2Zdxx); gpuFree(gd2Zdxx);
dd2Zdxy = gpuGetData(gd2Zdxy); gpuFree(gd2Zdxy);
d2Zd2y = gpuGetData(gd2Zdyy); gpuFree(gd2Zdyy);

gpuMem = gpuMem - gpuDeviceMemInfo()

assert_checkalmostequal(zz1, dzz1, 10*%eps, [], "matrix");
assert_checkalmostequal(dzpdx, ddzpdx, 10*%eps, [], "matrix");
assert_checkalmostequal(dzpdy, ddzpdy, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdxx, d2Zd2x, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdxy, dd2Zdxy, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdyy, d2Zd2y, 10*%eps, [], "matrix");
if gpuMem <> 0 then pause, end;


outmode = "natural";
gpuMem = gpuDeviceMemInfo();
[zz1 dzpdx dzpdy d2Zdxx d2Zdxy d2Zdyy]= interp2d(XX,YY, x, y, C, outmode);
[gzz1 gdzpdx gdzpdy gd2Zdxx gd2Zdxy gd2Zdyy] = gpuInterp2d(XX,YY, x, y, C, outmode);
dzz1 = gpuGetData(gzz1); gpuFree(gzz1);
ddzpdx = gpuGetData(gdzpdx); gpuFree(gdzpdx);
ddzpdy = gpuGetData(gdzpdy); gpuFree(gdzpdy);
d2Zd2x = gpuGetData(gd2Zdxx); gpuFree(gd2Zdxx);
dd2Zdxy = gpuGetData(gd2Zdxy); gpuFree(gd2Zdxy);
d2Zd2y = gpuGetData(gd2Zdyy); gpuFree(gd2Zdyy);

gpuMem = gpuMem - gpuDeviceMemInfo()

assert_checkalmostequal(zz1, dzz1, 10*%eps, [], "matrix");
assert_checkalmostequal(dzpdx, ddzpdx, 10*%eps, [], "matrix");
assert_checkalmostequal(dzpdy, ddzpdy, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdxx, d2Zd2x, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdxy, dd2Zdxy, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdyy, d2Zd2y, 10*%eps, [], "matrix");
if gpuMem <> 0 then pause, end;


outmode = "periodic";
gpuMem = gpuDeviceMemInfo();
[zz1 dzpdx dzpdy d2Zdxx d2Zdxy d2Zdyy]= interp2d(XX,YY, x, y, C, outmode);
[gzz1 gdzpdx gdzpdy gd2Zdxx gd2Zdxy gd2Zdyy] = gpuInterp2d(XX,YY, x, y, C, outmode);
dzz1 = gpuGetData(gzz1); gpuFree(gzz1);
ddzpdx = gpuGetData(gdzpdx); gpuFree(gdzpdx);
ddzpdy = gpuGetData(gdzpdy); gpuFree(gdzpdy);
d2Zd2x = gpuGetData(gd2Zdxx); gpuFree(gd2Zdxx);
dd2Zdxy = gpuGetData(gd2Zdxy); gpuFree(gd2Zdxy);
d2Zd2y = gpuGetData(gd2Zdyy); gpuFree(gd2Zdyy);

gpuMem = gpuMem - gpuDeviceMemInfo()

assert_checkalmostequal(zz1, dzz1, 10*%eps, [], "matrix");
assert_checkalmostequal(dzpdx, ddzpdx, 10*%eps, [], "matrix");
assert_checkalmostequal(dzpdy, ddzpdy, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdxx, d2Zd2x, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdxy, dd2Zdxy, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdyy, d2Zd2y, 10*%eps, [], "matrix");
if gpuMem <> 0 then pause, end;


outmode = "by_zero";
gpuMem = gpuDeviceMemInfo();
[zz1 dzpdx dzpdy d2Zdxx d2Zdxy d2Zdyy]= interp2d(XX,YY, x, y, C, outmode);
[gzz1 gdzpdx gdzpdy gd2Zdxx gd2Zdxy gd2Zdyy] = gpuInterp2d(XX,YY, x, y, C, outmode);
dzz1 = gpuGetData(gzz1); gpuFree(gzz1);
ddzpdx = gpuGetData(gdzpdx); gpuFree(gdzpdx);
ddzpdy = gpuGetData(gdzpdy); gpuFree(gdzpdy);
d2Zd2x = gpuGetData(gd2Zdxx); gpuFree(gd2Zdxx);
dd2Zdxy = gpuGetData(gd2Zdxy); gpuFree(gd2Zdxy);
d2Zd2y = gpuGetData(gd2Zdyy); gpuFree(gd2Zdyy);

gpuMem = gpuMem - gpuDeviceMemInfo()

assert_checkalmostequal(zz1, dzz1, 10*%eps, [], "matrix");
assert_checkalmostequal(dzpdx, ddzpdx, 10*%eps, [], "matrix");
assert_checkalmostequal(dzpdy, ddzpdy, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdxx, d2Zd2x, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdxy, dd2Zdxy, 10*%eps, [], "matrix");
assert_checkalmostequal(d2Zdyy, d2Zd2y, 10*%eps, [], "matrix");
if gpuMem <> 0 then pause, end;
