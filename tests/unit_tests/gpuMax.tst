// =============================================================================
// Scilab ( http://www.scilab.org/ ), This file is part of Scilab
// Copyright (C) DIGITEO, 2011, Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

// <-- ENGLISH IMPOSED -->

A=rand(3,2);
Z=rand(3,4)+%i*rand(3,4);

//------------ Real/Comple,------------
assert_checkequal(gpuMax(A), max(A));

z=gpuMax(Z);
assert_checkequal(abs(real(z)) + abs(imag(z)), max(abs(real(Z)) + abs(imag(Z))));

//------- Perform operation with gpuPointer------
da = gpuSetData(A);
assert_checkequal(gpuMax(da), max(A));


// elementwise Max
B  = rand(3,2);
ZB = rand(3,4)+%i*rand(3,4);

dres = gpuMax(A, B);
assert_checkequal(gpuGetData(dres), max(A, B));

dres = gpuMax(Z, ZB);
z = gpuGetData(dres);
assert_checkequal(abs(real(z)) + abs(imag(z)), max(abs(real(Z)) + abs(imag(Z)), abs(real(ZB)) + abs(imag(ZB))));
