// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2008 - INRIA - Allan CORNET
// Copyright (C) 2011 - DIGITEO - Allan CORNET
// Copyright (C) 2011 - DIGITEO - Cedric Delamarre
//
// This file is released under the 3-clause BSD license. See COPYING-BSD.

function subdemolist = demo_gateway()
  demopath = get_absolute_file_path("sciGPGPU.dem.gateway.sce");

  subdemolist = [   "D2Q9 CPU"                      ,"D2Q9/D2Q9.sce"; ..
                    "D2Q9 GPU"                      ,"D2Q9/D2Q9_gpu.sce"; ..
                    "Mandelbrot"                    ,"Mandelbrot/fract.sce"; ..
                    "Bench Transpose"               ,"matrix_transpose/transposeBench.sce"; ..
                    "Bench Transpose (two times)"   ,"matrix_transpose/transposeDoubleBench.sce";..
                    "Interpolation"                 ,"interpolation/interp.sce";..
                    "Interpolation 2D"              ,"interpolation/interp2d.sce";];

  subdemolist(:,2) = demopath + subdemolist(:,2);

endfunction

subdemolist = demo_gateway();
clear demo_gateway; // remove demo_gateway on stack
