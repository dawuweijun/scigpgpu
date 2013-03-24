// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) DIGITEO - 2011 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

gpuExit();
msg = lasterror();
if length(msg) <> 0 then pause, end
gpuInit();
msg = lasterror();
if length(msg) <> 0 then pause, end

