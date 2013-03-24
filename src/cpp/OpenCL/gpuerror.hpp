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

#ifndef ERROR_OPENCL_HPP_
#define ERROR_OPENCL_HPP_

#include <exception>
#include <iostream>
#include "Scierror.h"

template<>
inline int GpuError::treat_error<OPENCLmode>(OPENCLmode::Status id, int who)
{
	switch (id)
	{
	case CL_SUCCESS:
		return 0;
	case CL_DEVICE_NOT_FOUND:
		Scierror(id,"OpenCL error : Device Not Found.");
	case CL_DEVICE_NOT_AVAILABLE:
		Scierror(id,"OpenCL error : Device Not Available.");
	case CL_COMPILER_NOT_AVAILABLE:
		Scierror(id,"OpenCL error : Compiler Not Available.");
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		Scierror(id,"OpenCL error : Mem Object Allocation Failure.");
	case CL_OUT_OF_RESOURCES:
		Scierror(id,"OpenCL error : Out Of Ressources.");
	case CL_OUT_OF_HOST_MEMORY:
		Scierror(id,"OpenCL error : Out Of Host Memory.");
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		Scierror(id,"OpenCL error : Profiling Info Not Available.");
	case CL_MEM_COPY_OVERLAP:
		Scierror(id,"OpenCL error : Mem Copy Overlap.");
	case CL_IMAGE_FORMAT_MISMATCH:
		Scierror(id,"OpenCL error : Image Format Mismatch.");
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		Scierror(id,"OpenCL error : Image Format Not Supported.");
	case CL_BUILD_PROGRAM_FAILURE:
		Scierror(id,"OpenCL error : Build Program Failure.");
	case CL_MAP_FAILURE:
		Scierror(id,"OpenCL error : Map Failure.");
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		Scierror(id,"OpenCL error : Misaligned sub buffer offset.");
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		Scierror(id,"OpenCL error : Exec status error for events in wait list.");
	case CL_INVALID_VALUE:
		Scierror(id,"OpenCL error : Invalid Value.");
	case CL_INVALID_DEVICE_TYPE:
		Scierror(id,"OpenCL error : Invalid Device Type.");
	case CL_INVALID_PLATFORM:
		Scierror(id,"OpenCL error : Invalid Platform.");
	case CL_INVALID_DEVICE:
		Scierror(id,"OpenCL error : Invalid Device.");
	case CL_INVALID_CONTEXT:
		Scierror(id,"OpenCL error : Invalid Context.");
	case CL_INVALID_QUEUE_PROPERTIES:
		Scierror(id,"OpenCL error : Invalid Queue Properties.");
	case CL_INVALID_COMMAND_QUEUE:
		Scierror(id,"OpenCL error : Invalid Command Queue.");
	case CL_INVALID_HOST_PTR:
		Scierror(id,"OpenCL error : Invalid Host Ptr.");
	case CL_INVALID_MEM_OBJECT:
		Scierror(id,"OpenCL error : Invalid Mem Object.");
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		Scierror(id,"OpenCL error : Invalid Image Format Descriptor.");
	case CL_INVALID_IMAGE_SIZE:
		Scierror(id,"OpenCL error : Invalid Image Size.");
	case CL_INVALID_SAMPLER:
		Scierror(id,"OpenCL error : Invalid Sampler.");
	case CL_INVALID_BINARY:
		Scierror(id,"OpenCL error : Invalid Binary.");
	case CL_INVALID_BUILD_OPTIONS:
		Scierror(id,"OpenCL error : Invalid Build Options.");
	case CL_INVALID_PROGRAM:
		Scierror(id,"OpenCL error : Invalid Program.");
	case CL_INVALID_PROGRAM_EXECUTABLE:
		Scierror(id,"OpenCL error : Invalid Program Executable.");
	case CL_INVALID_KERNEL_NAME:
		Scierror(id,"OpenCL error : Invalid Kernel Name.");
	case CL_INVALID_KERNEL_DEFINITION:
		Scierror(id,"OpenCL error : Invalid Kernel Definition.");
	case CL_INVALID_KERNEL:
		Scierror(id,"OpenCL error : Invalid Kernel.");
	case CL_INVALID_ARG_INDEX:
		Scierror(id,"OpenCL error : Invalid Arg Index.");
	case CL_INVALID_ARG_VALUE:
		Scierror(id,"OpenCL error : Invalid Arg Value.");
	case CL_INVALID_KERNEL_ARGS:
		Scierror(id,"OpenCL error : Invalid Kernel Args.");
	case CL_INVALID_WORK_DIMENSION:
		Scierror(id,"OpenCL error : Invalid Work Dimension.");
	case CL_INVALID_WORK_GROUP_SIZE:
		Scierror(id,"OpenCL error : Invalid Work Group Size.");
	case CL_INVALID_WORK_ITEM_SIZE:
		Scierror(id,"OpenCL error : Invalid Work Item Size.");
	case CL_INVALID_GLOBAL_OFFSET:
		Scierror(id,"OpenCL error : Invalid Global Offset.");
	case CL_INVALID_EVENT_WAIT_LIST:
		Scierror(id,"OpenCL error : Invalid Event Wait List.");
	case CL_INVALID_EVENT:
		Scierror(id,"OpenCL error : Invalid Event.");
	case CL_INVALID_OPERATION:
		Scierror(id,"OpenCL error : Invalid Operation.");
	case CL_INVALID_GL_OBJECT:
		Scierror(id,"OpenCL error : Invalid GL Object.");
	case CL_INVALID_BUFFER_SIZE:
		Scierror(id,"OpenCL error : Invalid Buffer Size.");
	case CL_INVALID_MIP_LEVEL:
		Scierror(id,"OpenCL error : Invalid Mip Level.");
	case CL_INVALID_GLOBAL_WORK_SIZE:
		Scierror(id,"OpenCL error : Invalid Global Work Size.");
	default:
		break;
	}
    return -1;
}


#endif /* ERROR_H_ */
