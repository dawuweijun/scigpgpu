// ptrGpu(position) = ptrGpu
function result = %ptr_i_ptr(varargin)
    if isGpuPointer(varargin($)) & isGpuPointer(varargin($-1))
        [lhs,rhs]=argn(0);

        if rhs == 3
            if isGpuPointer(varargin($-2)) == %f & typeof(varargin($-2)) <> "constant"
                error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%ptr_i_ptr"));
            end

            // eye() or : => a(:)
            if size(varargin($-2)) == [-1 -1]
                position = 1:size(varargin($), '*');
            else
                position = varargin($-2);
            end

            gpuInsert(varargin($), varargin($-1), position);
            result = varargin($);

        elseif rhs == 4
            rows = varargin($-3);
            cols = varargin($-2);

            if isGpuPointer(rows) == %f & typeof(rows) <> "constant"
                error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%ptr_i_ptr"));
            end

            if isGpuPointer(cols) == %f & typeof(cols) <> "constant"
                error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%ptr_i_ptr"));
            end

            rsizes = size(rows);
            csizes = size(cols);

            // eye() or : => a(:,:)
            if rsizes == [-1 -1] & csizes == [-1 -1]
                result = gpuMatrix(varargin($), size(varargin($)));
                return;
            end

            // eye() or : => a(:,1)
            if rsizes == [-1 -1]
                rows = 1:size(varargin($), 'r');
            end

            // eye() or : => a(1,:)
            if csizes == [-1 -1]
                cols = 1:size(varargin($), 'c');
            end

            gpuRows = gpuMatrix(rows, -1, 1);
            gpuCols = gpuMatrix(cols, 1, -1);

            gpuInsert(varargin($), varargin($-1), gpuRows, gpuCols);

            gpuFree(gpuRows);
            gpuFree(gpuCols);

            result = varargin($);
        else
            error(msprintf(gettext("%s : Invalid index : 2D index expected.\n"), "%ptr_i_ptr"));
        end
    else
        error(msprintf(gettext("%s: Wrong type for input argument #1: GPU pointer expected."), "%ptr_i_ptr"));
    end
endfunction
