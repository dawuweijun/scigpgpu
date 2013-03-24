// =============================================================================
// Copyright (C) 2011 - DIGITEO - Allan CORNET
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
// This file is released into the public domain
// =============================================================================
mode(-1);
lines(0);
toolbox_dir = get_absolute_file_path("builder.sce");

function bOK = isNvidia()
    bOK = %f;
    if getos() == "Windows" then
        [dyninfo, statinfo] = getdebuginfo();
        version = getversion("scilab");
        if or(version(1:3) >= [5 4 0]) then
            videocard = dyninfo(grep(dyninfo, "Video card #"));
        else
            videocard = dyninfo(grep(dyninfo, "Video card:"));
            videocard = strsubst(videocard, "Video card:", "");
        end
    else
        videocard = unix_g("lspci | grep VGA");
    end

    bOK = grep(convstr(videocard, "u"), "NVIDIA") <> [];
endfunction

function bOK = isATI()
    bOK = %f;
    if getos() == "Windows" then
        [dyninfo, statinfo] = getdebuginfo();
        version = getversion("scilab");
        if or(version(1:3) >= [5 4 0]) then
            videocard = dyninfo(grep(dyninfo, "Video card #"));
        else
            videocard = dyninfo(grep(dyninfo, "Video card:"));
            videocard = strsubst(videocard, "Video card:", "");
        end
    else
        videocard = unix_g("lspci | grep VGA");
    end
    bOK = grep(convstr(videocard, "u"), "ATI") <> [];
endfunction

function bOK = generateConfigHeader(WITH_CUDA, WITH_OPENCL, toolbox_dir)
    header = [  "#ifndef __CONFIG_GPU_H__"; ..
                "#define __CONFIG_GPU_H__"];

    if WITH_CUDA then
        header = [  header;
                    "#define WITH_CUDA"];
    end

    if WITH_OPENCL then
        header = [  header;
                    "#define WITH_OPENCL"];
    end

    header = [  header;
                "#endif /* __CONFIG_GPU_H__ */"];

    bOK = mputl(header,toolbox_dir + "/src/c/config_gpu.h");
endfunction

function main_builder()

    TOOLBOX_NAME = "sciGPGPU";
    TOOLBOX_TITLE = "Toolbox SciGPGPU";

    // Check Scilab's version
    // =============================================================================
    try
        v = getversion("scilab");
    catch
        error(gettext("Scilab 5.3 or more is required."));
    end

    if v(2) < 3 then
    // new API in scilab 5.3
        error(gettext('Scilab 5.3 or more is required.'));
    end

    // Check modules_manager module availability
    // =============================================================================

    if ~isdef('tbx_build_loader') then
      error(msprintf(gettext('%s module not installed."), 'modules_manager'));
    end

    if ~isdef('WITH_CUDA') | ~isdef('WITH_OPENCL') then
        error('WITH_CUDA and WITH_OPENCL not defined.')
    end
    // =============================================================================
    if type(WITH_CUDA) <> 4 | type(WITH_OPENCL) <> 4 then
        error('Wrong value for WITH_CUDA or WITH_OPENCL boolean expected.')
    end
    // =============================================================================
    if ~WITH_CUDA & ~WITH_OPENCL then
        error('WITH_CUDA and WITH_OPENCL can not false at the same time.')
    end
    // =============================================================================

    // =============================================================================
    generateConfigHeader(WITH_CUDA, WITH_OPENCL,toolbox_dir);
    clear generateConfigHeader;
    // =============================================================================
    // Gathering of all platform dependant variable
    // =============================================================================
    //CUDA_INCLUDE_DIR=mfscanf(u,'%s');
    //NVIDIA_LIB_DIR=mfscanf(u,'%s');
    //CUDA_LIB_DIR=mfscanf(u,'%s');

    // Not required on Windows

    if getos() <> "Windows" then
        u = mopen(toolbox_dir+'cudaInc.txt','rt');
        i = 0;
        while meof(u) == 0,
            i = i + 1;
            inc(1,i) = mfscanf(u,'%s');
        end;
        NBR_INC_DIR = i - 1;
        mclose(u);

        if length(inc)==0 then
            error("Please add the path to your cuda include in cudaInc.txt");
        end;

        v = mopen(toolbox_dir+'cudaLib.txt', 'rt');
        i = 0;
        while meof(v) == 0,
            i = i+1;
            libs(1,i) = mfscanf(v,'%s');
        end;
        NBR_LIB_DIR = i - 1;
        mclose(v);
        if length(libs) == 0 then
            error("Please add the path to your cuda library in cudaLib.txt");
        end;

        w = mopen(toolbox_dir+'cudaOption.txt', 'rt');
        i=0;
        while meof(w) == 0
            i = i+1;
            opt(1,i) = mfscanf(w,'%s');
        end;
        // the last mfscanf return []
        NBR_OPT=i-1;
        if WITH_CUDA then
            opt(1,NBR_OPT+1) = " -lcuda";
            opt(1,NBR_OPT+2) = " -lcudart";
            opt(1,NBR_OPT+3) = " -lcufft";
            opt(1,NBR_OPT+4) = " -lcublas";
            NBR_OPT = NBR_OPT + 4;
        end
        if WITH_OPENCL then
            opt(1,NBR_OPT+1) = " -lOpenCL";
            NBR_OPT = NBR_OPT + 1;
        end
        mclose(w);
    end

    if getos() == "Windows" then
        if isNvidia() then
            pathCudaInc = getenv("CUDA_INC_PATH","");
            if pathCudaInc == "" then
                error("Please install CUDA Toolkit 3.2 to 4.0");
            end
            pathCudaLib = getenv("CUDA_LIB_PATH", "");
            if pathCudaLib == "" then
                error("Please install CUDA Toolkit 3.2 to 4.0");
            end
        else
            pathCudaInc = getenv("AMDAPPSDKROOT","");
            if pathCudaInc == "" then
                pathCudaInc = getenv("ATISTREAMSDKROOT", "");
            end
            if pathCudaInc == "" then
                error("Please install AMD APP SDK");
            end
        end
    end

    if getos() == "Linux" then

        // find includes dir
        if WITH_CUDA
            findFile = "cuda.h";
        else
            findFile = "CL/cl.h";
        end
        toolkitFind=%F;
        for i=1:NBR_INC_DIR,
            if isfile(inc(1,i)+filesep()+findFile) then
                toolkitFind=%T;
            end
        end
        if ~toolkitFind then
            error("Please install hardware Toolkit and put the include path in cudaInc.txt.");
        end

        // find libs dir
        libPath='';
        toolkitFind=%F;
        if WITH_CUDA
            findFile = "libcudart.so";
        else
            findFile = "libOpenCL.so";
        end
        for i=1:NBR_LIB_DIR,
            if isfile(libs(1,i)+filesep()+findFile) then
                toolkitFind=%T;
            end
        end
        if ~toolkitFind then
            error("Please install hardware Toolkit and put the library path in cudaLib.txt.");
        end
    end

    if getos() == "Darwin" then
        if ~isdir('/usr/local/cuda/include') then
            error("Please install CUDA Toolkit 3.2 to 4.0");
        end
        if ~isdir('/usr/local/cuda/lib') then
            error("Please install CUDA Toolkit 3.2 to 4.0");
        end
    end

    // Action & copy files needed when loading
    // =============================================================================
    tbx_builder_macros(toolbox_dir);
    tbx_builder_src(toolbox_dir);
    if WITH_CUDA
        mkdir(toolbox_dir+".libs/WithCuda/src/c");
        [s,m]=copyfile(toolbox_dir+"src/c/libgpuc"+ getdynlibext(), toolbox_dir+".libs/WithCuda/src/c/");
        if s == 0 then error("copyfile lib src c : "+m); end;
        mkdir(toolbox_dir+".libs/WithCuda/src/cpp");
        [s,m]=copyfile(toolbox_dir+"src/cpp/libgpucpp"+ getdynlibext(), toolbox_dir+".libs/WithCuda/src/cpp/");
        if s == 0 then error("copyfile lib src cpp : "+m); end;
        mkdir(toolbox_dir+".libs/WithCuda/src/cu");
        [s,m]=copyfile(toolbox_dir+"src/cu/libcudaKernels"+ getdynlibext(), toolbox_dir+".libs/WithCuda/src/cu/");
        if s == 0 then error("copyfile lib src cu : "+m); end;
    else
        mkdir(toolbox_dir+".libs/NoCuda/src/c");
        [s,m]=copyfile(toolbox_dir+"src/c/libgpuc"+ getdynlibext(), toolbox_dir+".libs/NoCuda/src/c/");
        if s == 0 then error("copyfile lib src c : "+m); end;
        mkdir(toolbox_dir+".libs/NoCuda/src/cpp");
        [s,m]=copyfile(toolbox_dir+"src/cpp/libgpucpp"+ getdynlibext(), toolbox_dir+".libs/NoCuda/src/cpp/");
        if s == 0 then error("copyfile lib src cpp : "+m); end;
    end

    tbx_builder_gateway(toolbox_dir);
    if WITH_CUDA
        mkdir(toolbox_dir+".libs/WithCuda/sci_gateway/c");
        [s,m]=copyfile(toolbox_dir+"sci_gateway/c/libgpu_c"+ getdynlibext(), toolbox_dir+".libs/WithCuda/sci_gateway/c/");
        if s == 0 then error("copyfile lib gateway c : "+m); end;
        [s,m]=copyfile(toolbox_dir+"sci_gateway/c/loader.sce", toolbox_dir+".libs/WithCuda/sci_gateway/c/");
        if s == 0 then error("copyfile loader c : "+m); end;
        mkdir(toolbox_dir+".libs/WithCuda/sci_gateway/cpp");
        [s,m]=copyfile(toolbox_dir+"sci_gateway/cpp/libgpgpu"+ getdynlibext(), toolbox_dir+".libs/WithCuda/sci_gateway/cpp/");
        if s == 0 then error("copyfile gateway cpp : "+m); end;
        [s,m]=copyfile(toolbox_dir+"sci_gateway/cpp/loader.sce", toolbox_dir+".libs/WithCuda/sci_gateway/cpp/");
        if s == 0 then error("copyfile loader cpp : "+m); end;
    else
        mkdir(toolbox_dir+".libs/NoCuda/sci_gateway/c");
        [s,m]=copyfile(toolbox_dir+"sci_gateway/c/libgpu_c"+ getdynlibext(), toolbox_dir+".libs/NoCuda/sci_gateway/c/");
        if s == 0 then error("copyfile lib gateway c : "+m); end;
        [s,m]=copyfile(toolbox_dir+"sci_gateway/c/loader.sce", toolbox_dir+".libs/NoCuda/sci_gateway/c/");
        if s == 0 then error("copyfile loader c : "+m); end;
        mkdir(toolbox_dir+".libs/NoCuda/sci_gateway/cpp");
        [s,m]=copyfile(toolbox_dir+"sci_gateway/cpp/libgpgpu"+ getdynlibext(), toolbox_dir+".libs/NoCuda/sci_gateway/cpp/");
        if s == 0 then error("copyfile gateway cpp : "+m); end;
        [s,m]=copyfile(toolbox_dir+"sci_gateway/cpp/loader.sce", toolbox_dir+".libs/NoCuda/sci_gateway/cpp/");
        if s == 0 then error("copyfile loader cpp : "+m); end;
    end

    tbx_builder_help(toolbox_dir);
    tbx_build_loader(TOOLBOX_NAME, toolbox_dir);
    tbx_build_cleaner(TOOLBOX_NAME, toolbox_dir);

endfunction

// =============================================================================
// Modify here by %T(true) or %F (false)
if isfile(toolbox_dir+filesep()+"cleaner.sce")
    exec(toolbox_dir+filesep()+"cleaner.sce")
end

WITH_CUDA = %T;
WITH_OPENCL = %T;

if isNvidia() == %F
    WITH_CUDA = %F;
end

if WITH_CUDA
    main_builder();
    WITH_CUDA = %F;
    if WITH_OPENCL
        exec(toolbox_dir+filesep()+"cleaner.sce")
    end
end

if WITH_OPENCL
    main_builder();
end

clear main_builder; // remove main_builder on stack
clear WITH_CUDA;
clear WITH_OPENCL;
clear toolbox_dir;
// =============================================================================

