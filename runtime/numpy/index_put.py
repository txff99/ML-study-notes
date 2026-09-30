from tensor import Tensor
import numpy as np

def index_put_numpy_impl(src: Tensor, shape_info, value: Tensor|np.ndarray):
    from util import get_default_strides
    # src data and value (if tensor) are both 1 dimentional.
    shape,stride,offset = shape_info
    if isinstance(value, Tensor):
        assert value.is_realized == True
        value = value.data
    old_data = src.data
    new_data = value
    for ptr in range(np.prod(shape)):
       # map each element in new data to old data
        indices = np.zeros(len(shape))
        rem = ptr
        # compute index for new data
        for i,s in enumerate(get_default_strides(shape)):
            indices[i] = 0 if s == 0 else rem // s
            if s != 0:  rem %= s
        old_data_ptr = np.dot(indices, src.strides).astype(int) + offset
        old_data[old_data_ptr] = new_data[ptr]
    return old_data