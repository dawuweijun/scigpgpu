// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) DIGITEO - 2011 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

// <-- ENGLISH IMPOSED -->

A = rand(3,2);
Z = rand(3,4) + %i * rand(3,4);

// ------------- Real/Complex -------------
assert_checkalmostequal(gpuSum(A), sum(A), %eps);

z = gpuSum(Z);
assert_checkalmostequal(abs(real(z)) + abs(imag(z)), sum(abs(real(Z)) + abs(imag(Z))), %eps);


