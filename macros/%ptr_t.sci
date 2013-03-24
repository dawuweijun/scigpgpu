function ptrTransposed = %ptr_t(ptr_value)
    if isGpuPointer(ptr_value) then ptrTransposed = gpuTranspose(ptr_value), end;
endfunction
