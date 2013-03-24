// =============================================================================
// Scilab ( http://www.scilab.org/ ), This file is part of Scilab
// Copyright (C) DIGITEO, 2011, Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

// <-- ENGLISH IMPOSED -->

A = rand(3,2);
Z = rand(3,4) + %i * rand(3,4);

//------------ Real/Complex------------
assert_checkequal(gpuMin(A), min(A));

z=gpuMin(Z);
assert_checkequal(abs(real(z)) + abs(imag(z)), min(abs(real(Z)) + abs(imag(Z))));

//------- Perform operation with gpuPointer------
da = gpuSetData(A);
assert_checkequal(gpuMin(da), min(A));

// elementwise Min
B  = rand(3,2);
ZB = rand(3,4)+%i*rand(3,4);

dres = gpuMin(A, B);
assert_checkequal(gpuGetData(dres), min(A, B));

dres = gpuMin(Z, ZB);
z = gpuGetData(dres);
assert_checkequal(abs(real(z)) + abs(imag(z)), min(abs(real(Z)) + abs(imag(Z)), abs(real(ZB)) + abs(imag(ZB))));
