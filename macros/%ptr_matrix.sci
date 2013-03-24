function res = %ptr_matrix(varargin)
    if isGpuPointer(varargin(1))
        res = gpuMatrix(varargin(:))
    else
        error(msprintf(gettext("%s: Wrong type for input argument #1: GPU pointer expected."), "matrix"));
    end
endfunction
