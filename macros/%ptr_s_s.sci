function res = %ptr_s_s(left, right)
    if isGpuPointer(left)
        res = gpuSubtract(left, right);
    else
        error(msprintf(gettext("%s: Wrong type for input arguments: GPU pointer expected."), "subtract"));
    end
endfunction
