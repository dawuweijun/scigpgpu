// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) DIGITEO - 2011 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

_dir = pwd();
chdir(TMPDIR);

if gpuUseCuda() then
	copyfile(gpgpu_getToolboxPath() + "/tests/unit_tests"+"/matrixAdd.cu",TMPDIR+"/matrixAdd.cu");
end
if ~gpuUseCuda() then
	copyfile(gpgpu_getToolboxPath() + "/tests/unit_tests"+"/matrixAdd.cl",TMPDIR+"/matrixAdd.cl");
end

gpuBuild(TMPDIR+"/matrixAdd");

if gpuUseCuda() then
	if ~isfile(TMPDIR+"/matrixAdd.ptx") then pause,end
end

if ~gpuUseCuda() then
	if ~isfile(TMPDIR+"/matrixAdd.cl.out") then pause,end
end

cd(_dir);
clear _dir;
