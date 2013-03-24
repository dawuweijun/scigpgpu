function res = %s_s_ptr(left, right)
    if isGpuPointer(right)
        res = gpuSubtract(left, right);
    else
        error(msprintf(gettext("%s: Wrong type for input arguments: GPU pointer expected."), "subtract"));
    end
endfunction
