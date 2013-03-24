/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
* Copyright (C) DIGITEO - 2010 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#ifndef ERROR_CUDA_HPP_
#define ERROR_CUDA_HPP_

#include <exception>
#include <iostream>
#include <cublas.h>
#include <cufft.h>
#include "Scierror.h"
#include <cuda_runtime_api.h>

#define CUDA 0
#define CUBLAS 1
#define CUFFT 2

template<>
inline
int GpuError::treat_error<CUDAmode>(CUDAmode::Status id, int who)
{
    char errorMessage[500];

    switch( who )
    {
    case CUDA:
        switch( id )
        {
            //for api cu... (ie : cuMemAlloc)
            case CUDA_SUCCESS:
            return 0;
            case CUDA_ERROR_INVALID_VALUE:
                Scierror(id,"Cuda error : One or more of the parameters passed to the API call is not within an acceptable range of values.\n");break;
            case CUDA_ERROR_OUT_OF_MEMORY:
                Scierror(id,"Cuda error : The API call failed because it was unable to allocate enough memory to perform the requested operation.\n");break;
            case CUDA_ERROR_NOT_INITIALIZED:
                Scierror(id,"Cuda error : CUDA driver has not been initialized with cuInit() or that initialization has failed.\n");break;
            case CUDA_ERROR_DEINITIALIZED:
                Scierror(id,"Cuda error : CUDA driver is in the process of shutting down.\n");break;
            case CUDA_ERROR_NO_DEVICE:
                Scierror(id,"Cuda error : no CUDA-capable devices were detected by the installed CUDA driver.\n");break;
            case CUDA_ERROR_INVALID_DEVICE:
                Scierror(id,"Cuda error : the device ordinal supplied by the user does not correspond to a valid CUDA device.\n");break;
            case CUDA_ERROR_INVALID_IMAGE:
                Scierror(id,"Cuda error : the device kernel image is invalid. This can also indicate an invalid CUDA module.\n");break;
            case CUDA_ERROR_INVALID_CONTEXT:
                Scierror(id,"Cuda error : there is no context bound to the current thread.\n");break;
            case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
                Scierror(id,"Cuda error : the context being supplied as a parameter to the API call was already the active context.\n");break;
            case CUDA_ERROR_MAP_FAILED:
                Scierror(id,"Cuda error : map or register operation has failed.\n");break;
            case CUDA_ERROR_UNMAP_FAILED:
                Scierror(id,"Cuda error : unmap or unregister operation has failed.\n");break;
            case CUDA_ERROR_ARRAY_IS_MAPPED:
                Scierror(id,"Cuda error : specified array is currently mapped and thus cannot be destroyed.\n");break;
            case CUDA_ERROR_ALREADY_MAPPED:
                Scierror(id,"Cuda error : resource is already mapped.\n");break;
            case CUDA_ERROR_NO_BINARY_FOR_GPU:
                Scierror(id,"Cuda error : there is no kernel image available that is suitable for the device.\n");break;
            case CUDA_ERROR_ALREADY_ACQUIRED:
                Scierror(id,"Cuda error : resource has already been acquired.\n");break;
            case CUDA_ERROR_NOT_MAPPED:
                Scierror(id,"Cuda error : resource is not mapped.\n");break;
            case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
                Scierror(id,"Cuda error : mapped resource is not available for access as an array.\n");break;
            case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
                Scierror(id,"Cuda error : mapped resource is not available for access as a pointer.\n");break;
            case CUDA_ERROR_ECC_UNCORRECTABLE:
                Scierror(id,"Cuda error : uncorrectable ECC error was detected during execution.\n");break;
            #ifdef CUDA_EROOR_UNSUPPORTED_LIMIT
            case CUDA_ERROR_UNSUPPORTED_LIMIT:
                Scierror(id,"Cuda error : the ::CUlimit passed to the API call is not supported by the active device.\n");break;
            #endif
            case CUDA_ERROR_INVALID_SOURCE:
                Scierror(id,"Cuda error : the device kernel source is invalid.\n");break;
            case CUDA_ERROR_FILE_NOT_FOUND:
                Scierror(id,"Cuda error : the file specified was not found.\n");break;
            #ifdef CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND
            case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
                Scierror(id, "Cuda error : link to a shared object failed to resolve.\n");break;
            #endif
            case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
                Scierror(id, "Cuda error : initialization of a shared object failed.\n");break;
            case CUDA_ERROR_OPERATING_SYSTEM:
                Scierror(id, "Cuda error : OS call failed.\n");break;
            case CUDA_ERROR_INVALID_HANDLE:
                Scierror(id, "Cuda error : resource handle passed to the API call was not valid. Resource handles are opaque types like ::CUstream and ::CUevent.\n");break;
            case CUDA_ERROR_NOT_FOUND:
                Scierror(id, "Cuda error : named symbol was not found. Examples of symbols are global/constant variable names, texture names, and surface names.\n");break;
            case CUDA_ERROR_NOT_READY:
                Scierror(id, "Cuda error : asynchronous operations issued previously have not completed yet.\n");break;
            case CUDA_ERROR_LAUNCH_FAILED:
                Scierror(id, "Cuda error : An exception occurred on the device while executing a kernel.\n");break;
            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
                Scierror(id, "Cuda error : launch did not occur because it did not have appropriate resources.\n");break;
            case CUDA_ERROR_LAUNCH_TIMEOUT:
                Scierror(id, "Cuda error : the device kernel took too long to execute.\n");break;
            case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
                Scierror(id, "Cuda error : kernel launch that uses an incompatible texturing mode.\n");break;
            #ifdef  CUDA_ERROR_POINTER_IS_64BIT
            case CUDA_ERROR_POINTER_IS_64BIT:
                Scierror(id, "Cuda error : pointer is 64 bit.\n");break;
            #endif
            #ifdef CUDA_ERROR_SIZE_IS_64BIT
            case CUDA_ERROR_SIZE_IS_64BIT:
                Scierror(id, "Cuda error : Size is 64 bit.\n");break;
            #endif
            case CUDA_ERROR_UNKNOWN:
                Scierror(id,"Cuda error : unknown internal error has occurred.\n");break;

            //for api cuda... (ie : cudaAlloc)
            case cudaErrorPriorLaunchFailure :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorPriorLaunchFailure));Scierror(id,errorMessage);break;
            case cudaErrorLaunchTimeout :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorLaunchTimeout));Scierror(id,errorMessage);break;
            case cudaErrorLaunchOutOfResources :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorLaunchOutOfResources));Scierror(id,errorMessage);break;
            case cudaErrorInvalidDeviceFunction :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidDeviceFunction));Scierror(id,errorMessage);break;
            case cudaErrorInvalidConfiguration :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidConfiguration));Scierror(id,errorMessage);break;
            case cudaErrorInvalidDevice :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidDevice));Scierror(id,errorMessage);break;
            case cudaErrorInvalidValue :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidValue));Scierror(id,errorMessage);break;
            case cudaErrorInvalidPitchValue :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidPitchValue));Scierror(id,errorMessage);break;
            case cudaErrorInvalidSymbol :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidSymbol));Scierror(id,errorMessage);break;
            case cudaErrorMapBufferObjectFailed :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorMapBufferObjectFailed));Scierror(id,errorMessage);break;
            case cudaErrorUnmapBufferObjectFailed :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorUnmapBufferObjectFailed));Scierror(id,errorMessage);break;
            case cudaErrorInvalidHostPointer :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidHostPointer));Scierror(id,errorMessage);break;
            case cudaErrorInvalidDevicePointer :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidDevicePointer));Scierror(id,errorMessage);break;
            case cudaErrorInvalidTexture :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidTexture));Scierror(id,errorMessage);break;
            case cudaErrorInvalidTextureBinding :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidTextureBinding));Scierror(id,errorMessage);break;
            case cudaErrorInvalidChannelDescriptor :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidChannelDescriptor));Scierror(id,errorMessage);break;
            case cudaErrorInvalidMemcpyDirection :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidMemcpyDirection));Scierror(id,errorMessage);break;
            case cudaErrorAddressOfConstant :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorAddressOfConstant));Scierror(id,errorMessage);break;
            case cudaErrorTextureFetchFailed :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorTextureFetchFailed));Scierror(id,errorMessage);break;
            case cudaErrorTextureNotBound :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorTextureNotBound));Scierror(id,errorMessage);break;
            case cudaErrorSynchronizationError :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorSynchronizationError));Scierror(id,errorMessage);break;
            case cudaErrorInvalidFilterSetting :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidFilterSetting));Scierror(id,errorMessage);break;
            case cudaErrorInvalidNormSetting :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidNormSetting));Scierror(id,errorMessage);break;
            case cudaErrorMixedDeviceExecution :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorMixedDeviceExecution));Scierror(id,errorMessage);break;
            case cudaErrorCudartUnloading :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorCudartUnloading));Scierror(id,errorMessage);break;
            case cudaErrorUnknown :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorUnknown));Scierror(id,errorMessage);break;
            case cudaErrorNotYetImplemented :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorNotYetImplemented));Scierror(id,errorMessage);break;
            case cudaErrorMemoryValueTooLarge :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorMemoryValueTooLarge));Scierror(id,errorMessage);break;
            case cudaErrorInvalidResourceHandle :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidResourceHandle));Scierror(id,errorMessage);break;
            case cudaErrorNotReady :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorNotReady));Scierror(id,errorMessage);break;
            case cudaErrorInsufficientDriver :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInsufficientDriver));Scierror(id,errorMessage);break;
            case cudaErrorSetOnActiveProcess :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorSetOnActiveProcess));Scierror(id,errorMessage);break;
            case cudaErrorInvalidSurface :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidSurface));Scierror(id,errorMessage);break;
            case cudaErrorNoDevice :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorNoDevice));Scierror(id,errorMessage);break;
            case cudaErrorECCUncorrectable :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorECCUncorrectable));Scierror(id,errorMessage);break;
            case cudaErrorSharedObjectSymbolNotFound :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorSharedObjectSymbolNotFound));Scierror(id,errorMessage);break;
            case cudaErrorSharedObjectInitFailed :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorSharedObjectInitFailed));Scierror(id,errorMessage);break;
            case cudaErrorUnsupportedLimit :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorUnsupportedLimit));Scierror(id,errorMessage);break;
            case cudaErrorDuplicateVariableName :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorDuplicateVariableName));Scierror(id,errorMessage);break;
            case cudaErrorDuplicateTextureName :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorDuplicateTextureName));Scierror(id,errorMessage);break;
            case cudaErrorDuplicateSurfaceName :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorDuplicateSurfaceName));Scierror(id,errorMessage);break;
            case cudaErrorDevicesUnavailable :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorDevicesUnavailable));Scierror(id,errorMessage);break;
            case cudaErrorInvalidKernelImage :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorInvalidKernelImage));Scierror(id,errorMessage);break;
            case cudaErrorNoKernelImageForDevice :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorNoKernelImageForDevice));Scierror(id,errorMessage);break;
            case cudaErrorIncompatibleDriverContext :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorIncompatibleDriverContext));Scierror(id,errorMessage);break;
            case cudaErrorStartupFailure :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorStartupFailure));Scierror(id,errorMessage);break;
            case cudaErrorApiFailureBase :
                sprintf(errorMessage,"Cuda error : %s\n",cudaGetErrorString(cudaErrorApiFailureBase));Scierror(id,errorMessage);break;
            default:
                Scierror(id,"Not Cuda Error.\n");
        }
    break;
    case CUBLAS:
        switch ( id )
        {
            case CUBLAS_STATUS_SUCCESS :
            return 0;
            case CUBLAS_STATUS_NOT_INITIALIZED :
                Scierror(id,"Cublas error : CUBLAS library not initialized.\n");break;
            case CUBLAS_STATUS_ALLOC_FAILED :
                Scierror(id,"Cublas error :resource allocation failed.\n");break;
            case CUBLAS_STATUS_INVALID_VALUE :
                Scierror(id,"Cublas error : unsupported numerical value was passed to function.\n");break;
            case CUBLAS_STATUS_ARCH_MISMATCH :
                Scierror(id,"Cublas error : function requires an architectural feature absent from the architecture of the device.\n");break;
            case CUBLAS_STATUS_MAPPING_ERROR :
                Scierror(id,"Cublas error : access to GPU memory space failed.\n");break;
            case CUBLAS_STATUS_EXECUTION_FAILED :
                Scierror(id,"Cublas error : GPU program failed to execute.\n");break;
            case CUBLAS_STATUS_INTERNAL_ERROR :
                Scierror(id,"Cublas error : an internal CUBLAS operation failed.\n");break;
            default :
                Scierror(id,"Not Cublas Error.\n");
        }
    break;
    case CUFFT:
        switch ( id )
        {
            case CUFFT_SUCCESS :
            return 0;
            case CUFFT_INVALID_PLAN :
                Scierror(id,"Cufft error : CUFFT is passed an invalid plan handle.\n");break;
            case CUFFT_ALLOC_FAILED :
                Scierror(id,"Cufft error : CUFFT failed to allocate GPU memory.\n");break;
            case CUFFT_INVALID_TYPE :
                Scierror(id,"Cufft error : The user requests an unsupported type.\n");break;
            case CUFFT_INVALID_VALUE :
                Scierror(id,"Cufft error : The user specifies a bad memory pointer.\n");break;
            case CUFFT_INTERNAL_ERROR :
                Scierror(id,"Cufft error : Used for all internal driver errors.\n");break;
            case CUFFT_EXEC_FAILED :
                Scierror(id,"Cufft error : CUFFT failed to execute an FFT on the GPU.\n");break;
            case CUFFT_SETUP_FAILED :
                Scierror(id,"Cufft error : The CUFFT library failed to initialize.\n");break;
            case CUFFT_INVALID_SIZE :
                Scierror(id,"Cufft error : The user specifies an unsupported FFT size.\n");break;
            case CUFFT_UNALIGNED_DATA :
                Scierror(id,"Cufft error : Input or output does not satisfy texture alignment requirements.\n");break;
            default :
                Scierror(id,"Not cufft Error.\n");
        }
    break;
    default :
        Scierror(999,"Not GPU Error.\n");
    }

    return -1;
}


#endif /* ERROR_H_ */
