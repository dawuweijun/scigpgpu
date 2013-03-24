function result=%ptr_x_s(left, right)
    if isGpuPointer(left)
        result = gpuDotMult(left, right);
    end
endfunction
