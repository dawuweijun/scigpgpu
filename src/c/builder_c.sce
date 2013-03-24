// ====================================================================
// Copyright (C) 2011 - DIGITEO - Allan CORNET
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
// This file is released into the public domain
// ====================================================================
// This file is released under the 3-clause BSD license. See COPYING-BSD.
// ====================================================================

function builder_src_c()

    src_c_path = get_absolute_file_path("builder_c.sce");
    // ====================================================================
    CFLAGS = "-I" + src_c_path;
    // ====================================================================
    files_c = ["useCuda.c", ..
               "with_cuda.c", ..
               "with_opencl.c"];
    // ====================================================================           
    tbx_build_src([	"gpuc"],  ..
                  files_c,    ..
                  "c",        ..
                  src_c_path, ..
                  "",         ..
                  "",         ..
                  CFLAGS);
endfunction

builder_src_c();
clear builder_src_c; // remove builder_src_c on stack

