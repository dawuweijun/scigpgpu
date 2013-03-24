function result=%ptr_a_s(left, right)
    if isGpuPointer(left)
        result = gpuAdd(left, right);
    end
endfunction
