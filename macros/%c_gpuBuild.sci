function bin = %c_gpuBuild(cufile,varargin)

    rhs = argn(2);
    opt = "";

    if rhs == 2 then
        opt = varargin(1);
    end

    if ~isfile(cufile+".cu");
        error(999,cufile+".cu not found. " );
    end

    if ~isdef("NVCC_PATH") then
        getNVCC_PATH();
    end

    if getos() == "Windows" then
        commandLine = """" + NVCC_PATH + '\nvcc.exe"" -arch sm_13 -ptx -o "+fullpath(cufile)+".ptx '  + '""' + cufile+ ".cu" + '""'
    else
        commandLine = NVCC_PATH + '/nvcc -arch sm_13 -ptx -o "+fullpath(cufile)+".ptx ' + cufile + ".cu"
    end

    [rep,stat,stderr] = unix_g(commandLine + " " +opt);

    if stat then
        mprintf("%s\n",stderr);
        error(999,"Error when building with NVCC ");
    end

    bin = [cufile+".ptx";"Cuda"]
    NVCC_PATH = resume(NVCC_PATH);

endfunction
