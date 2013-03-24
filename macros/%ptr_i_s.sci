// -------------- warning ---------------------
// This overload send all variables in device
// memory, perform the insertion in device
// and return the result in host memory.
// H(2) = 3, all of H is send to device and
// return in host memory.
// --------------------------------------------

// constant(position) = ptrGpu
function result = %ptr_i_s(varargin)
disp(varargin)
    if isGpuPointer(varargin($-1))

        [lhs,rhs]=argn(0);
        da = gpuSetData(varargin($));

        select rhs
        case 3 then
            if isGpuPointer(varargin($-2)) == %f & typeof(varargin($-2)) <> "constant"
                error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%ptr_i_s"));
            end
            da(varargin(1)) = varargin(2);
        case 4 then
            rows = varargin($-3);
            cols = varargin($-2);
            if isGpuPointer(rows) == %f & typeof(rows) <> "constant"
                error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%ptr_i_s"));
            end

            if isGpuPointer(cols) == %f & typeof(cols) <> "constant"
                error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%ptr_i_s"));
            end

            da(rows, cols) = varargin(3);
        else
            error(msprintf(gettext("%s : Invalid index : 2D index expected.\n"), "%ptr_i_s"));
        end

        result = gpuGetData(da);
        da = gpuFree(da);
    else
        error(msprintf(gettext("%s: Wrong type for input argument #1: GPU pointer expected."), "%ptr_i_s"));
    end
endfunction
