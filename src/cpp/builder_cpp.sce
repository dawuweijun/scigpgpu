// ====================================================================
// Copyright (C) 2011 - DIGITEO - Allan CORNET
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
// This file is released into the public domain
// ====================================================================
// This file is released under the 3-clause BSD license. See COPYING-BSD.

function builder_src_cpp()

    src_cpp_path = get_absolute_file_path("builder_cpp.sce");
    src_c_path = fullpath(src_cpp_path + "/../c");

    CFLAGS = "-I" + src_cpp_path;
    CFLAGS = CFLAGS + " -I" + src_c_path;

    if WITH_CUDA then
        src_cu_path = fullpath(src_cpp_path + "/../cu");
        CFLAGS = CFLAGS + " -I" + src_cu_path;
    end

    LDFLAGS  = "";

    if getos() == "Windows" then
        CFLAGS = CFLAGS + ' -DGPU_EXPORTS ';

        if isNvidia() then
            pathCudaInc = getenv("CUDA_INC_PATH","");
            if(pathCudaInc <> "")
                CFLAGS = CFLAGS + ' -I""' + pathCudaInc +'"" ';
            end

            pathCudaLib = getenv("CUDA_LIB_PATH", "");
            if(pathCudaInc <> "")
                LDFLAGS = "/LIBPATH:""" + pathCudaLib + """";
            end
        else
            pathATIAPPRoot = getenv("ATISTREAMSDKROOT", "");
            if pathATIAPPRoot == ""
                pathATIAPPRoot = getenv("AMDAPPSDKROOT", "");
            end

            pathOpenCLInclude = fullpath(pathATIAPPRoot + "include");
            pathOpenCLLib = fullpath(pathATIAPPRoot + "lib");
            CFLAGS = CFLAGS + ' -I""' + pathOpenCLInclude + '"" ';

            if win64() then
                LDFLAGS = "/LIBPATH:""" + pathOpenCLLib + "/x86_64"+ """";
            else
                LDFLAGS = "/LIBPATH:""" + pathOpenCLLib + "/x86" + """";
            end
        end
    end

    if getos() == "Linux" then
        for i=1:NBR_INC_DIR,
            CFLAGS = CFLAGS + ' -I'+inc(1,i);
        end

        LDFLAGS = " -L/usr/lib ";
        for i=1:NBR_LIB_DIR,
            LDFLAGS = LDFLAGS+' -L'+libs(1,i);
        end

        if WITH_CUDA then
            LDFLAGS = LDFLAGS + " -lcudart -lcuda ";
        end

        if WITH_OPENCL then
            LDFLAGS = LDFLAGS + " -lOpenCL "
        end
    end

    if getos() <> "Windows" then
        CFLAGS = CFLAGS + " -std=c++0x ";
    end

    if getos() == "Darwin" then
        CFLAGS = CFLAGS + ' -I' + '/System/Library/Frameworks/OpenGL.framework/Headers'
        if WITH_OPENCL then
            CFLAGS = CFLAGS + ' -I' + '/System/Library/Frameworks/OpenCL.framework/Headers';
        end
    end

    files_cpp = ["checkDevice.cpp",      ..
                "gpuPointerManager.cpp", ..
                "gpuContext.cpp",        ..
                "gpuSplin2d.cpp",        ..
                "gpuKronecker.cpp"];
    lib_dependencies = "../c/libgpuc";

    if WITH_CUDA then
        files_cpp = [files_cpp,  "pointerCuda.cpp", ..
                                 "cudaElemMin.cpp", ..
                                 "cudaElemMax.cpp", ..
                                 "cudaDotMult.cpp", ..
                                 "cudaRealImgToComplex.cpp"];
        lib_dependencies = [lib_dependencies, "../cu/libcudakernels"];
    end

    if WITH_OPENCL then
        files_cpp = [files_cpp, "pointerOpenCL.cpp", "builderOpenCL.cpp"];
    end

    if getos() == "Windows" then
        files_cpp = [files_cpp, "DllMainGPGPU.c"];
    end

    tbx_build_src(  ["gpucpp"],             ..
                    files_cpp,              ..
                    "cpp",                  ..
                    src_cpp_path,           ..
                    lib_dependencies,       ..
                    LDFLAGS,                ..
                    CFLAGS);
endfunction

builder_src_cpp();
clear builder_src_cpp; // remove builder_src_cpp on stack

