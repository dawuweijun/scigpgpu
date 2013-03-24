function result = %ptr_e(varargin)
    if isGpuPointer(varargin($))
        [lhs,rhs]=argn(0);

        if rhs == 2
            if isGpuPointer(varargin($-1)) == %f & typeof(varargin($-1)) <> "constant"
                error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%ptr_e"));
            end

            // eye() or : => a(:)
            if size(varargin($-1)) == [-1 -1]
                result = gpuMatrix(varargin($), -1, 1);
                return;
            end

            position = varargin($-1);
            result = gpuExtract(varargin($), position);

        elseif rhs == 3
            rows = varargin($-2);
            cols = varargin($-1);

            if isGpuPointer(rows) == %f & typeof(rows) <> "constant"
                    error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%ptr_e"));
            end

            if isGpuPointer(cols) == %f & typeof(cols) <> "constant"
                    error(msprintf(gettext("%s: Wrong type of index: A constant or gpu variable expected."), "%ptr_e"));
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

            result = gpuExtract(varargin($), gpuRows, gpuCols);

            gpuFree(gpuRows)
            gpuFree(gpuCols)
        else
            error(msprintf(gettext("%s : Invalid index : 2D index expected.\n"), "%ptr_e"));
        end
    else
        error(msprintf(gettext("%s: Wrong type for input argument #1: GPU pointer expected."), "%ptr_e"));
    end
endfunction
