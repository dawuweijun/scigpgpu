function result=%ptr_k_ptr(left, right)
    if isGpuPointer(left) & isGpuPointer(right)
        result = gpuKronecker(left, right);
    end
endfunction
