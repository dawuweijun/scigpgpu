function result=%ptr_m_s(left, right)
    if isGpuPointer(left)
        result = gpuMult(left, right);
    end
endfunction
