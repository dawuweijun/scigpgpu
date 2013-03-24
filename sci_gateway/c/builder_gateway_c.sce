// ====================================================================
// Copyright (C) 2011 - DIGITEO - Allan CORNET
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
// This file is released into the public domain
// ====================================================================

function builder_gw_c()
    if getos() == "Windows" then
        // to manage long pathname
        includes_src_c = "-I""" + get_absolute_file_path("builder_gateway_c.sce") + "../../src/c""";
    else
        includes_src_c = "-I" + get_absolute_file_path("builder_gateway_c.sce") + "../../src/c";
    end
    // ====================================================================
    // PutLhsVar managed by user in sci_sum and in sci_sub
    // if you do not this variable, PutLhsVar is added
    // in gateway generated (default mode in scilab 4.x and 5.x)
    WITHOUT_AUTO_PUTLHSVAR = %t;
    // ====================================================================
    gw_files = ["sci_gpuWithCuda.c", ..
                "sci_gpuWithOpenCL.c"];

    gw_tables = ["gpuWithCuda", "sci_gpuWithCuda";
                 "gpuWithOpenCL", "sci_gpuWithOpenCL"];

    // ====================================================================
    lib_dependencies = ["../../src/c/libgpuc"];

    // ====================================================================
    tbx_build_gateway(  "libgpu_c", ..
                        gw_tables, ..
                        gw_files, ..
                        get_absolute_file_path("builder_gateway_c.sce"), ..
                        lib_dependencies, ..
                        "", ..
                        includes_src_c);
endfunction

builder_gw_c();
clear builder_gw_c; // remove builder_gw_c on stack

