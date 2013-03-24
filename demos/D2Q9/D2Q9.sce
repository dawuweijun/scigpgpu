// Copyright (C) 2010 - DIGITEO - Vincent Lejeune
// Copyright (C) 2011 - DIGITEO - Cedric Delamarre
//
// This file must be used under the terms of the CeCILL.
// This source file is licensed as described in the file COPYING, which
// you should have received as part of this distribution.  The terms
// are also available at
// http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt

lines(0);

abs_path=get_absolute_file_path("D2Q9.sce");
funcprot(0);
exec(abs_path+"scibench_latticeboltz.sci");

// GENERAL FLOW CONSTANTS
lx     = 400;      //number of cells in x-direction
ly     = 100;      // number of cells in y-direction
maxT   = 400;  // total number of iterations
tPlot  = 1;//50;      // cycles

scibench_latticeboltz ( lx , ly , maxT , tPlot )

