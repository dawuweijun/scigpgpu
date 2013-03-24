// gpuPtr(position) = constant
function result = %s_i_ptr(varargin)
    if isGpuPointer(varargin($))

        [lhs,rhs]=argn(0);
        result = varargin($);
        da = gpuSetData(varargin($-1));

        select rhs
        case 3 then
            if isGpuPointer(varargin(1)) == %f & typeof(varargin(1)) <> "constant"
                error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%s_i_ptr"));
            end

            result(varargin(1)) = da;
            gpuFree(da)
        case 4 then
            rows = varargin(1);
            cols = varargin(2);
            if isGpuPointer(rows) == %f & typeof(rows) <> "constant"
                error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%s_i_ptr"));
            end

            if isGpuPointer(cols) == %f & typeof(cols) <> "constant"
                error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%s_i_ptr"));
            end

            result(rows, cols) = da;
            gpuFree(da)
        else
            error(msprintf(gettext("%s : Invalid index : 2D index expected.\n"), "%s_i_ptr"));
        end

    else
        error(msprintf(gettext("%s: Wrong type for input argument #1: GPU pointer expected."), "%s_i_ptr"));
    end
endfunction
