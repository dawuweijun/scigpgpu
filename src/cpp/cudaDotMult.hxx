/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) Scilab Enterprises - 2013 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/
#ifndef __CUDADOTMULT_HXX__
#define __CUDADOTMULT_HXX__

#include "pointerCuda.hxx"
#include "gpuContext.hxx"
#include "dynlib_gpu.h"

GPU_IMPEXP PointerCuda* cudaDotMult(PointerCuda* gpuPtrA, PointerCuda* gpuPtrB);

#endif //__CUDADOTMULT_HXX__
