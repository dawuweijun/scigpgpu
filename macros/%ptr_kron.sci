function kronRes = %ptr_kron(varargin)
    [lhs,rhs]=argn(0);

    if rhs > 2
        error(msprintf(gettext("%s: Wrong size of input argument: %d to %d expected."), "kron", 1, 2));
    end

    if isGpuPointer(varargin(1)) == %f & typeof(varargin(1)) <> "constant" then
        error(msprintf(gettext("%s: Wrong type for input argument #1: GPU pointer expected."), "kron"));
    end

    if rhs == 2 & isGpuPointer(varargin(2)) == %f & typeof(varargin(2)) <> "constant" then
        error(msprintf(gettext("%s: Wrong type for input argument #2: GPU pointer expected."), "kron"));
    end

    kronRes = gpuKronecker(varargin(:))

endfunction
