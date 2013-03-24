// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) DIGITEO - 2011 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================


before=gpuDeviceMemInfo();
d=gpuAlloc(1000,512);
d=gpuFree(d);
after=gpuDeviceMemInfo();

if before <> after then pause, end

a=rand(500,600);
before=gpuDeviceMemInfo();
da=gpuSetData(a);
da=gpuFree(da);
after=gpuDeviceMemInfo();

if before <> after then pause, end

