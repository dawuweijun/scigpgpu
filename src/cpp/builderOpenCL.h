/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
* 				 
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at    
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#include "gpu_wrap.h"
#include "opencl_wrap.h"

#include "dynlib_gpu.h"

GPU_IMPEXP class Builder: public Context<OPENCLmode>
{
	typedef Context<OPENCLmode> Base;

public:
	GPU_IMPEXP int build(char* sourcefile, std::string buildoption);
};

