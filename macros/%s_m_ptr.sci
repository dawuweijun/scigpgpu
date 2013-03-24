function result=%s_m_ptr(left, right)
    if isGpuPointer(right)
        result = gpuMult(left, right);
    end
endfunction
