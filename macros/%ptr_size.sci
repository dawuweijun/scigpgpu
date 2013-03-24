function varargout = %ptr_size(varargin)
    if isGpuPointer(varargin(1))
        [lhs,rhs]=argn(0);
        sizes = gpuSize(varargin(1));
        select lhs
        case 1 then
            if rhs == 2 then
                select varargin(2)
                case 'r' then
                    sizes = sizes(1);
                case 'c' then
                    sizes = sizes(2);
                case '*' then
                    sizes = prod(sizes);
                else
                    error(msprintf(gettext("%s: Wrong value for input argument #2: ''r'', ''c'' or ''*'' expected."), "size"));
                end
            end
            varargout = list(sizes);
        case 2 then
            if rhs <> 1
                error(msprintf(gettext("%s: Wrong size of input arguments: %d expected."), "size", 1));
            end
            varargout = list(sizes(1), sizes(2))
        else
            error(msprintf(gettext("%s: Wrong size of output arguments: %d to %d expected."), "size", 1, 2));
        end
    else
        error(msprintf(gettext("%s: Wrong type for input argument #1: GPU pointer expected."), "size"));
    end
endfunction
