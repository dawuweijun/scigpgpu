function result=%ptr_a_ptr(left, right)
    if isGpuPointer(left) & isGpuPointer(right)
        result = gpuAdd(left, right);
    end
endfunction
