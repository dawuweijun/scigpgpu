function result=%s_a_ptr(left, right)
    if isGpuPointer(right)
        result = gpuAdd(left, right);
    end
endfunction
