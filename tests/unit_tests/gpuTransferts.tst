// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) DIGITEO - 2011 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

A=rand(8,8);
gA=gpuSetData(A);
B=gpuGetData(gA);

if A <> B then pause,end

