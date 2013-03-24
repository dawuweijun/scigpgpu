function result=%ptr_x_ptr(left, right)
    if isGpuPointer(left) & isGpuPointer(right)
        result = gpuDotMult(left, right);
    end
endfunction
