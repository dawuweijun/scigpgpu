function result=%s_x_ptr(left, right)
    if isGpuPointer(right)
        result = gpuDotMult(left, right);
    end
endfunction
