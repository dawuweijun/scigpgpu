function result=%ptr_m_ptr(left, right)
    if isGpuPointer(left) & isGpuPointer(right)
        result = gpuMult(left, right);
    end
endfunction
