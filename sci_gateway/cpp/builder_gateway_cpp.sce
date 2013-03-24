// ====================================================================
// Copyright (C) 2011 - DIGITEO - Allan CORNET
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
// This file is released into the public domain
// ====================================================================

function builder_gw_cpp()

    gateway_cpp_path = get_absolute_file_path('builder_gateway_cpp.sce');
    // ====================================================================
    CFLAGS = " -I" +gateway_cpp_path;
    CFLAGS = CFLAGS + " -I" + fullpath(gateway_cpp_path + "../../src/c");
    CFLAGS = CFLAGS + " -I" + fullpath(gateway_cpp_path + "../../src/cpp");
    if WITH_CUDA then
        CFLAGS = CFLAGS + " -I" + fullpath(gateway_cpp_path + "../../src/cu");
    end

    LDFLAGS  = "";
    // ====================================================================
    if getos() == "Windows" then

        if isNvidia() then
            pathCudaInc = getenv("CUDA_INC_PATH","");
            CFLAGS = CFLAGS + ' -I""' + fullpath(pathCudaInc) + '"" ';

            pathCudaLib = getenv("CUDA_LIB_PATH", "");
            LDFLAGS = "/LIBPATH:""" + fullpath(pathCudaLib) + """";
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

        LDFLAGS = LDFLAGS + " /LIBPATH:""" + fullpath(gateway_cpp_path + "../../src/cpp") + """";
    else
        for i=1:NBR_INC_DIR,
            CFLAGS = CFLAGS+' -I'+inc(1,i);
        end

        for i=1:NBR_LIB_DIR,
            LDFLAGS = LDFLAGS+' -L'+libs(1,i);
        end

        for i=1:NBR_OPT,
            LDFLAGS = LDFLAGS+' '+opt(1,i);
        end

        if getos() == "Darwin" then

            CFLAGS = CFLAGS + ' -I' + '/System/Library/Frameworks/OpenGL.framework/Headers'
            if WITH_OPENCL then
                CFLAGS = CFLAGS + ' -I' + '/System/Library/Frameworks/OpenCL.framework/Headers';
            end
        end
    end

    // ====================================================================
    // PutLhsVar managed by user in sci_sum and in sci_sub
    // if you do not this variable, PutLhsVar is added
    // in gateway generated (default mode in scilab 4.x and 5.x)
    WITHOUT_AUTO_PUTLHSVAR = %t;

    // ====================================================================
    files_sci_gateway = [   "sci_gpuBuild.cpp",..
                            "sci_gpuInit.cpp", ..
                            "sci_gpuExit.cpp", ..
                            "sci_gpuAlloc.cpp", ..
                            "sci_gpuFree.cpp", ..
                            "sci_gpuApplyFunction.cpp", ..
                            "sci_gpuLoadFunction.cpp", ..
                            "sci_gpuDeviceInfo.cpp", ..
                            "sci_gpuDeviceMemInfo.cpp", ..
                            "sci_gpuDoubleCapability.cpp", ..
                            "sci_gpuSetData.cpp", ..
                            "sci_gpuGetData.cpp", ..
                            "sci_gpuAdd.cpp", ..
                            "sci_gpuMult.cpp", ..
                            "sci_gpuFFT.cpp", ..
                            "sci_gpuMax.cpp", ..
                            "sci_gpuMin.cpp", ..
                            "sci_gpuNorm.cpp", ..
                            "sci_gpuSum.cpp", ..
                            "sci_gpuTranspose.cpp", ..
                            "sci_gpuSize.cpp", ..
                            "sci_gpuUseCuda.cpp", ..
                            "sci_gpuPtrInfo.cpp", ..
                            "sci_isGpuPointer.cpp", ..
                            "sci_gpuInterp.cpp", ..
                            "sci_gpuInterp2d.cpp", ..
                            "sci_gpuMatrix.cpp", ..
                            "sci_gpuExtract.cpp", ..
                            "sci_gpuInsert.cpp", ..
                            "sci_gpuSubtract.cpp", ..
                            "sci_gpuClone.cpp", ..
                            "sci_gpuDotMult.cpp", ..
                            "sci_gpuComplex.cpp", ..
                            "sci_gpuSplin2d.cpp", ..
                            "sci_gpuKronecker.cpp", ..
                            "sci_gpuOnes.cpp", ..
                            "deviceInfo.cpp"];

    if getos() == "Windows" then
       files_sci_gateway = [files_sci_gateway, "DllMainGPGPU.c"];
    end
    // ====================================================================
    gw_table = ["gpuBuild",                 "sci_gpuBuild";..
                "gpuAdd",                   "sci_gpuAdd"; ..
                "gpuMult",                  "sci_gpuMult"; ..
                "gpuFFT",                   "sci_gpuFFT"; ..
                "gpuMax",                   "sci_gpuMax"; ..
                "gpuMin",                   "sci_gpuMin"; ..
                "gpuNorm",                  "sci_gpuNorm"; ..
                "gpuSum",                   "sci_gpuSum"; ..
                "gpuTranspose",             "sci_gpuTranspose"; ..
                "gpuAlloc",                 "sci_gpuAlloc"; ..
                "gpuApplyFunction",         "sci_gpuApplyFunction"; ..
                "gpuDeviceInfo",            "sci_gpuDeviceInfo"; ..
                "gpuDeviceMemInfo",         "sci_gpuDeviceMemInfo"; ..
                "gpuDoubleCapability",      "sci_gpuDoubleCapability"; ..
                "gpuExit",                  "sci_gpuExit"; ..
                "gpuFree",                  "sci_gpuFree"; ..
                "gpuGetData",               "sci_gpuGetData"; ..
                "gpuSetData",               "sci_gpuSetData"; ..
                "gpuInit",                  "sci_gpuInit"; ..
                "gpuLoadFunction",          "sci_gpuLoadFunction"; ..
                "gpuSize",                  "sci_gpuSize"; ..
                "gpuUseCuda",               "sci_gpuUseCuda"; ..
                "gpuPtrInfo",               "sci_gpuPtrInfo"; ..
                "isGpuPointer",             "sci_isGpuPointer"; ..
                "gpuInterp",                "sci_gpuInterp"; ..
                "gpuInterp2d",              "sci_gpuInterp2d"; ..
                "gpuMatrix",                "sci_gpuMatrix"; ..
                "gpuExtract",               "sci_gpuExtract"; ..
                "gpuInsert",                "sci_gpuInsert"; ..
                "gpuSubtract",              "sci_gpuSubtract"; ..
                "gpuClone",                 "sci_gpuClone"; ..
                "gpuDotMult",               "sci_gpuDotMult"; ..
                "gpuComplex",               "sci_gpuComplex"; ..
                "gpuSplin2d",               "sci_gpuSplin2d"; ..
                "gpuKronecker",             "sci_gpuKronecker"; ..
                "gpuOnes",                  "sci_gpuOnes"; ..
                ];

    // ====================================================================
    lib_dependencies = "../../src/c/libgpuc";
    if WITH_CUDA then
        lib_dependencies = [lib_dependencies, "../../src/cu/libcudaKernels"];
    end
    lib_dependencies = [lib_dependencies , "../../src/cpp/libgpucpp"];


    // ====================================================================

    CFLAGS = CFLAGS + " -std=c++0x ";

    tbx_build_gateway('libgpgpu', ..
                       gw_table, ..
                       files_sci_gateway, ..
                       gateway_cpp_path, ..
                       lib_dependencies, ..
                       LDFLAGS, CFLAGS);

endfunction

builder_gw_cpp();
clear builder_gw_cpp; // remove builder_gw_cpp on stack

