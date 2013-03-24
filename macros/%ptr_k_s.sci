function result=%ptr_k_s(left, right)
    if isGpuPointer(left)
        result = gpuKronecker(left, right);
    end
endfunction
