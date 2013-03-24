function %ptr_p(ptr_value)
    if isGpuPointer(ptr_value) then disp(gpuPtrInfo(ptr_value)), end;
endfunction
