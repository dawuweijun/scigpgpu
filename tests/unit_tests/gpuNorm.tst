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

if gpuNorm(A) - norm(A,'fro') <> 0 then pause, end

res = gpuNorm(Z) - norm(Z,'fro');
if imag(res) > 1D-15 then pause, end
if real(res) > 1D-15 then pause, end
