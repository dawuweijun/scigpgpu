// ====================================================================
// Copyright (C) 2011 - DIGITEO - Allan CORNET
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
// This file is released into the public domain
// ====================================================================
// This file is released under the 3-clause BSD license. See COPYING-BSD.

function builder_src_cu()

    src_cu_path = get_absolute_file_path("builder_cu.sce");
    src_c_path = fullpath(src_cu_path + "/../c");

    CFLAGS = " -I" + src_cu_path;
    CFLAGS = CFLAGS + " -I" + src_c_path;
    LDFLAGS = "";

    if getos() == "Windows" then
        pathCudaInc = getenv("CUDA_INC_PATH","");
        CFLAGS = CFLAGS + ' -I""' + pathCudaInc +'"" ';

        pathCudaLib = getenv("CUDA_LIB_PATH", "");
        LDFLAGS = "/LIBPATH:""" + pathCudaLib + """";
    end

    if getos() == "Linux" then
        for i=1:NBR_INC_DIR,    CFLAGS = CFLAGS+' -I'+inc(1,i); end;
        for i=1:NBR_LIB_DIR,    LDFLAGS = LDFLAGS+' -L'+libs(1,i);end;
        for i=1:NBR_OPT,        LDFLAGS = LDFLAGS+' '+opt(1,i);   end;
    end

    if getos() == "Darwin" then
        CFLAGS = CFLAGS + ' -I' + '/usr/local/cuda/include';
        LDFLAGS = LDFLAGS + ' -L' + '/usr/local/cuda/lib';
    end

    mprintf("Build Cuda files...");

    CudaFiles = "makecucomplex.cu "     + ..
                "dsum.cu "              + ..
                "zsum.cu "              + ..
                "idmin.cu "             + ..
                "idmax.cu "             + ..
                "elemWiseMin.cu "       + ..
                "elemWiseMax.cu "       + ..
                "interp.cu "            + ..
                "interp2d.cu "          + ..
                "extract.cu "           + ..
                "insert.cu "            + ..
                "initCudaMatrix.cu "    + ..
                "dotmult.cu "           + ..
                "splin2d.cu "           + ..
                "strictIncreasing.cu "  + ..
                "kronecker.cu "         + ..
                "matrixTranspose.cu ";

    oldPath = pwd();
    cd(src_cu_path);

    if getos() == "Windows" then
        CUDA_BIN_PATH = getenv("CUDA_BIN_PATH", "");
        [rep, stat, stderr] = unix_g(""""+getshortpathname(CUDA_BIN_PATH) + filesep() +'nvcc"" -g -arch sm_13 -cuda -I '+ src_c_path + ' ' + CudaFiles);
    end

    if getos() <> "Windows" then
        path = mopen(src_cu_path+'../../nvccdir.txt','rt');
        NVCC_PATH = mfscanf(path,'%s');
        mclose(path);
    end

    if getos() == "Linux" then
        [rep, stat, stderr] = unix_g(NVCC_PATH + '/nvcc -g -arch sm_13 -cuda -I '+ src_c_path + ' ' + CudaFiles);
    end

    if getos() == "Darwin" then
        [rep, stat, stderr] = unix_g(NVCC_PATH + '/nvcc -g -arch sm_13 -cuda -I '+ src_c_path + ' ' + CudaFiles);
    end

//    if getos() == "Windows" then
//        MV_CMD = "move";
//    else
//        MV_CMD = "mv";
//    end

// cuda > 4.1
//    unix_g(MV_CMD + " makecucomplex.cu.cpp.ii makecucomplex.cu.cpp");
//    unix_g(MV_CMD + " dsum.cu.cpp.ii dsum.cu.cpp");
//    unix_g(MV_CMD + " zsum.cu.cpp.ii zsum.cu.cpp");
//    unix_g(MV_CMD + " idmin.cu.cpp.ii idmin.cu.cpp");
//    unix_g(MV_CMD + " idmax.cu.cpp.ii idmax.cu.cpp");
//    unix_g(MV_CMD + " elemWiseMin.cu.cpp.ii elemWiseMin.cu.cpp");
//    unix_g(MV_CMD + " elemWiseMax.cu.cpp.ii elemWiseMax.cu.cpp");
//    unix_g(MV_CMD + " interp.cu.cpp.ii interp.cu.cpp");
//    unix_g(MV_CMD + " interp2d.cu.cpp.ii interp2d.cu.cpp");
//    unix_g(MV_CMD + " extract.cu.cpp.ii extract.cu.cpp");
//    unix_g(MV_CMD + " insert.cu.cpp.ii insert.cu.cpp");
//    unix_g(MV_CMD + " initCudaMatrix.cu.cpp.ii initCudaMatrix.cu.cpp");
//    unix_g(MV_CMD + " dotmult.cu.cpp.ii dotmult.cu.cpp");
//    unix_g(MV_CMD + " matrixTranspose.cu.cpp.ii matrixTranspose.cu.cpp");
//    unix_g(MV_CMD + " splin2d.cu.cpp.ii splin2d.cu.cpp");
//    unix_g(MV_CMD + " strictIncreasing.cu.cpp.ii strictIncreasing.cu.cpp");
//    unix_g(MV_CMD + " kronecker.cu.cpp.ii kronecker.cu.cpp");

    cd(oldPath);

    if stat then
        mprintf("failed \n");
        mprintf("%s\n",stderr);
        error(999,"Build Cuda failed !   ");
    else
        mprintf("Ok \n");
    end;

    files_cu_cpp = ["makecucomplex.cu.cpp",     ..
                    "dsum.cu.cpp",              ..
                    "zsum.cu.cpp",              ..
                    "idmin.cu.cpp",             ..
                    "idmax.cu.cpp",             ..
                    "elemWiseMin.cu.cpp",       ..
                    "elemWiseMax.cu.cpp",       ..
                    "interp.cu.cpp",            ..
                    "interp2d.cu.cpp",          ..
                    "extract.cu.cpp",           ..
                    "insert.cu.cpp",            ..
                    "initCudaMatrix.cu.cpp",    ..
                    "dotmult.cu.cpp",           ..
                    "splin2d.cu.cpp",           ..
                    "strictIncreasing.cu.cpp",  ..
                    "kronecker.cu.cpp",         ..
                    "matrixTranspose.cu.cpp"]

    if getos() == "Windows" then
        files_cu_cpp = [files_cu_cpp, "DllMainGPGPU.c"];
    end

    tbx_build_src(  ["cudaKernels"],    ..
                    files_cu_cpp,       ..
                    "c",                ..
                    src_cu_path,        ..
                    "", LDFLAGS, CFLAGS);

endfunction

builder_src_cu();
clear builder_src_cu; // remove builder_src_cu on stack

