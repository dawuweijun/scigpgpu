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
d = gpuTranspose(A);
h = gpuGetData(d);
if ~and(h == A') then pause, end
d = gpuFree(d);

d = gpuTranspose(Z);
h = gpuGetData(d);
if ~and(h == Z') then pause, end
d = gpuFree(d);
