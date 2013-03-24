function result=%s_k_ptr(left, right)
    if isGpuPointer(right)
        result = gpuKronecker(left, right);
    end
endfunction
